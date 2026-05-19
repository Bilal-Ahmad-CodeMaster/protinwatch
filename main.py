# main.py
"""
ProteinWatch — FastAPI Backend
-------------------------------
Viral protein sequence analysis pipeline:
  Layer 1 → K-mer novelty scoring        (kmer_compare)
  Layer 2 → ESM-2 ML danger scoring      (esm2_scorer)
  Layer 3 → AlphaFold structure lookup   (structure_compare)
  Layer 4 → Gemini generative brief      (gemini_brief)
  Layer 5 → Simulated alert dispatch     (simulate_action)

Deployment targets
  • Hugging Face Spaces (Docker, port 7860)  ← recommended free tier
  • Koyeb free tier (512 MB RAM)
  • Any ASGI-compatible host

Memory budget
  • Result cache   : LRU, max 100 entries  (~few KB each)
  • Geo cache      : LRU, max 200 entries
  • ESM-2 model    : lazy-loaded on first /analyze request
"""

import asyncio
import hashlib
import logging
import os
from collections import OrderedDict
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from geopy.geocoders import Nominatim

load_dotenv()

# ---------------------------------------------------------------------------
# Application modules
# ---------------------------------------------------------------------------
from esm2_scorer import danger_score, cache_info as esm2_cache_info
from gemini_brief import generate_brief_streaming
from kmer_compare import compute_novelty
from scheduler import scheduler, update_schedule
from simulate_action import simulate_alert_dispatch
from structure_compare import fetch_alphafold_structure, foldseek_search  # noqa: F401

import chromadb

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("proteinwatch")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULT_CACHE_MAXSIZE: int = int(os.getenv("RESULT_CACHE_MAXSIZE", "100"))
GEO_CACHE_MAXSIZE:    int = int(os.getenv("GEO_CACHE_MAXSIZE",    "200"))
DEFAULT_LAT:          float = float(os.getenv("DEFAULT_LAT", "29.3957"))  # Bahawalpur
DEFAULT_LNG:          float = float(os.getenv("DEFAULT_LNG", "71.6833"))
THREAT_ALERT_THRESHOLD: float = float(os.getenv("THREAT_ALERT_THRESHOLD", "60"))
AUTO_FETCH_INTERVAL:  str = os.getenv("AUTO_FETCH_INTERVAL", "30min")
CHROMA_PATH:          str = os.getenv("CHROMA_PATH", "data/chromadb")

# Virus metadata (UniProt IDs for AlphaFold + 3-D viewer)
VIRUS_METADATA: dict[str, dict] = {
    "SARS-CoV-2_Omicron_BA.5": {"pdb_id": "P0DTC2"},
    "Ebola_Virus":              {"pdb_id": "Q05320"},
    "Zika_Virus":               {"pdb_id": "Q32ZE1"},
    "Dengue":                   {"pdb_id": "P27909"},
    "Chikungunya":              {"pdb_id": "Q8JUX5"},
}
DEFAULT_PDB_ID = "P0DTC2"

# ---------------------------------------------------------------------------
# LRU result cache  (replaces unbounded dict)
# ---------------------------------------------------------------------------
class _LRUCache(OrderedDict):
    """Thread-safe-ish LRU dict with a fixed capacity."""

    def __init__(self, maxsize: int = 100):
        self._maxsize = maxsize
        super().__init__()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self._maxsize:
            oldest = next(iter(self))
            logger.debug("Result cache evicting key: %s", oldest)
            del self[oldest]

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value


result_cache: _LRUCache = _LRUCache(maxsize=RESULT_CACHE_MAXSIZE)
last_fetch_time: Optional[str] = None
chroma_col = None

# ---------------------------------------------------------------------------
# Geocoder  (lru_cache replaces manual geo_cache dict)
# ---------------------------------------------------------------------------
geolocator = Nominatim(user_agent="proteinwatch_app", timeout=5)


@lru_cache(maxsize=GEO_CACHE_MAXSIZE)
def _geocode(location_text: str) -> tuple[float, float]:
    """
    Resolve a free-text location to (lat, lng).
    Results are cached by lru_cache — evicted LRU when GEO_CACHE_MAXSIZE is hit.
    Returns the Bahawalpur default on failure.
    """
    if not location_text or location_text.lower() == "unknown":
        return DEFAULT_LAT, DEFAULT_LNG

    try:
        clean = location_text.replace(":", ",")
        result = geolocator.geocode(clean)
        if result:
            return result.latitude, result.longitude
    except Exception as exc:
        logger.warning("Geocoding error for '%s': %s", location_text, exc)

    return DEFAULT_LAT, DEFAULT_LNG


def get_lat_lng(location_text: str) -> dict:
    lat, lng = _geocode(location_text)
    return {"lat": lat, "lng": lng}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ProteinWatch API",
    description="Viral protein sequence analysis — novelty, danger & structure.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    global chroma_col

    os.makedirs("tmp", exist_ok=True)
    os.makedirs("data/structures", exist_ok=True)

    # ChromaDB — persistent vector store for viral sequences
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_col = client.get_or_create_collection("viral_sequences")

    # Background scheduler — auto-fetches new sequences
    scheduler.start()
    update_schedule(AUTO_FETCH_INTERVAL)

    logger.info(
        "🚀 ProteinWatch started. ChromaDB: %d sequences. "
        "Auto-fetch every %s.",
        chroma_col.count(),
        AUTO_FETCH_INTERVAL,
    )


@app.on_event("shutdown")
async def on_shutdown():
    if scheduler.running:
        scheduler.shutdown(wait=False)
    logger.info("ProteinWatch shut down cleanly.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
async def health():
    """Liveness + readiness probe."""
    return {
        "status":        "ok",
        "db_count":      chroma_col.count() if chroma_col else 0,
        "last_fetch":    last_fetch_time,
        "result_cache":  {"size": len(result_cache), "maxsize": RESULT_CACHE_MAXSIZE},
        "esm2_cache":    esm2_cache_info(),
    }


@app.post("/analyze", tags=["Analysis"])
async def analyze(body: dict):
    """
    Full analysis pipeline for a viral protein sequence.

    Body fields
    -----------
    sequence      : str   — amino-acid string (≥ 50 chars)
    location_text : str   — optional free-text location for geocoding
    """
    sequence: str = body.get("sequence", "").strip()

    if len(sequence) < 50:
        return JSONResponse({"error": "Sequence too short (minimum 50 residues)."}, 400)

    # Cache hit → return immediately, no compute needed
    seq_hash = hashlib.md5(sequence.encode()).hexdigest()
    if seq_hash in result_cache:
        logger.debug("Result cache hit: %s", seq_hash[:8])
        return result_cache[seq_hash]

    # --- Layer 1 + 2 in parallel ---
    kmer_result, esm2_result = await asyncio.gather(
        asyncio.to_thread(compute_novelty, sequence),
        asyncio.to_thread(danger_score, sequence),
    )

    matched_virus: str = kmer_result.get("closest_match", "Unknown")

    # --- Geocoding ---
    location_text: str = body.get("location_text") or matched_virus
    coords = await asyncio.to_thread(get_lat_lng, location_text)

    # --- Layer 3: AlphaFold structure ---
    meta = VIRUS_METADATA.get(matched_virus, {"pdb_id": DEFAULT_PDB_ID})
    struct_path = await asyncio.to_thread(fetch_alphafold_structure, meta["pdb_id"])

    # --- Threat index ---
    kmer_score:  float = kmer_result.get("novelty_score", 50.0)
    esm2_score:  float = esm2_result.get("danger_score",  50.0)
    struct_score: float = 82.0 if struct_path else 70.0

    threat_index: float = round(
        kmer_score  * 0.25
        + esm2_score  * 0.45
        + struct_score * 0.30,
        1,
    )

    result = {
        "analysis_id":      seq_hash[:8],
        "threat_index":     threat_index,
        "kmer_score":       kmer_score,
        "esm2_score":       esm2_score,
        "structural_score": struct_score,
        "closest_match":    matched_virus,
        "lat":              coords["lat"],
        "lng":              coords["lng"],
        "pdb_id":           meta["pdb_id"],
        "alert":            None,
    }

    # --- Layer 5: auto-dispatch alert ---
    if threat_index > THREAT_ALERT_THRESHOLD:
        result["alert"] = simulate_alert_dispatch(
            seq_hash[:8], threat_index, matched_virus
        )

    result_cache[seq_hash] = result
    return result


@app.get("/stream-brief", tags=["Analysis"])
async def stream_brief(
    sequence: str,
    threat_index: float = 50.0,
    kmer: float = 50.0,
    esm2: float = 50.0,
):
    """Server-sent events stream of the Gemini-generated threat brief."""
    scores = {"combined": threat_index, "kmer": kmer, "esm2": esm2}

    async def _generator():
        try:
            for chunk in generate_brief_streaming(sequence, scores):
                yield f"data: {chunk}\n\n"
        except Exception as exc:
            logger.error("Brief generation error: %s", exc)
            yield f"data: Error generating brief: {exc}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generator(), media_type="text/event-stream")


@app.post("/simulate-action", tags=["Actions"])
async def simulate_action(body: dict):
    """Manually trigger a simulated alert dispatch."""
    return simulate_alert_dispatch(
        body.get("sequence_id",  "unknown"),
        body.get("threat_index", 75.0),
        body.get("virus_name",   "Unknown Virus"),
    )


@app.post("/scheduler/update", tags=["System"])
async def update_sched(body: dict):
    """Update the auto-fetch interval (e.g. '30min', '1h', '6h')."""
    return update_schedule(body.get("label", "6h"))


@app.get("/history", tags=["Analysis"])
async def history(limit: int = 50):
    """Return cached analyses sorted by threat index (highest first)."""
    sorted_results = sorted(
        result_cache.values(),
        key=lambda x: x.get("threat_index", 0),
        reverse=True,
    )
    return sorted_results[:limit]


@app.get("/structure/{uniprot_id}", tags=["Structure"])
async def get_structure(uniprot_id: str):
    """Return the raw PDB file for a given UniProt ID (if cached locally)."""
    path = f"data/structures/{uniprot_id}.pdb"
    if os.path.exists(path):
        with open(path) as fh:
            return {"pdb": fh.read()}
    return JSONResponse({"error": f"Structure '{uniprot_id}' not found."}, 404)


@app.get("/agent-trace/{analysis_id}", tags=["Actions"])
async def get_trace(analysis_id: str):
    """Return the agent reasoning trace for a previously analysed sequence."""
    for result in result_cache.values():
        if result.get("analysis_id") == analysis_id:
            alert = result.get("alert") or {}
            return alert.get("agent_trace", [])
    return []


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def _global_error_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url, exc)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "fallback": "Using cached data"},
    )