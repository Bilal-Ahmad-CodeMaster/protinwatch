# main.py
"""
ProteinWatch — FastAPI Backend
-------------------------------
Fully Dynamic Viral protein sequence analysis pipeline:
  Layer 1 → K-mer novelty scoring        (kmer_compare)
  Layer 2 → ESM-2 ML danger scoring      (esm2_scorer)
  Layer 3 → Dynamic AI Metadata Fallback (Groq API for Location & PDB)
  Layer 4 → AlphaFold / ESMFold lookup   (structure_compare)
  Layer 5 → Generative brief             (gemini_brief / Groq)
  Layer 6 → Simulated alert dispatch     (simulate_action)

FIXES APPLIED:
  - CUSTOM_FOLD: pLDDT extracted from ESMFold PDB (no hardcoded 95.0).
  - AlphaFold structures also scored via pLDDT (no hardcoded 82.0).
  - Geocoding: Groq location string is cleaned before passing to Nominatim.
    If Nominatim still fails, a second Groq call returns lat/lng directly.
  - Bahawalpur fallback only fires when all geo attempts genuinely fail.
"""

import asyncio
import hashlib
import logging
import os
import re
import json
from collections import OrderedDict
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from geopy.geocoders import Nominatim
from groq import Groq
import chromadb

load_dotenv()

# ---------------------------------------------------------------------------
# Application modules
# ---------------------------------------------------------------------------
from esm2_scorer import danger_score, cache_info as esm2_cache_info
from gemini_brief import generate_brief_streaming
from kmer_compare import compute_novelty
from scheduler import scheduler, update_schedule
from simulate_action import simulate_alert_dispatch
from structure_compare import (
    fetch_alphafold_structure,
    foldseek_search,
    fold_with_esmfold,
    compute_structural_score,
)

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
RESULT_CACHE_MAXSIZE:   int   = int(os.getenv("RESULT_CACHE_MAXSIZE", "100"))
GEO_CACHE_MAXSIZE:      int   = int(os.getenv("GEO_CACHE_MAXSIZE",    "200"))
DEFAULT_LAT:            float = float(os.getenv("DEFAULT_LAT", "29.3957"))   # Bahawalpur fallback
DEFAULT_LNG:            float = float(os.getenv("DEFAULT_LNG", "71.6833"))
THREAT_ALERT_THRESHOLD: float = float(os.getenv("THREAT_ALERT_THRESHOLD", "60"))
AUTO_FETCH_INTERVAL:    str   = os.getenv("AUTO_FETCH_INTERVAL", "30min")
CHROMA_PATH:            str   = os.getenv("CHROMA_PATH", "data/chromadb")

# Known virus → UniProt/AlphaFold IDs (extend as needed)
VIRUS_METADATA: dict[str, dict] = {
    "SARS-CoV-2_Omicron_BA.5": {"pdb_id": "P0DTC2"},
    "Ebola_Virus":              {"pdb_id": "Q05320"},
    "Zika_Virus":               {"pdb_id": "Q32ZE1"},
    "Dengue":                   {"pdb_id": "P27909"},
    "Chikungunya":              {"pdb_id": "Q8JUX5"},
}

# ---------------------------------------------------------------------------
# Groq helpers
# ---------------------------------------------------------------------------
def _groq_client() -> Optional[Groq]:
    key = os.environ.get("GROQ_API_KEY", "")
    return Groq(api_key=key) if key else None


def fetch_dynamic_virus_info(virus_name: str) -> dict:
    """
    Ask Groq for the virus origin location and AlphaFold/UniProt accession.
    Returns a clean human-readable location string (e.g. "Wuhan, China")
    and a PDB/UniProt ID (e.g. "P0DTC2") or None.
    """
    if not virus_name or virus_name.lower() == "unknown":
        return {"location": "Unknown", "pdb_id": None}

    client = _groq_client()
    if not client:
        return {"location": virus_name, "pdb_id": None}

    try:
        prompt = (
            f"Provide information for the virus: '{virus_name}'.\n"
            "Reply ONLY in valid JSON with exactly two keys:\n"
            '1. "location": city and country where this virus was first discovered '
            '(e.g. "Wuhan, China" or "Zaire, DRC"). '
            "Must be a real geocodable place name — NOT a virus name or strain label.\n"
            '2. "pdb_id": 6-character UniProt accession of its main surface protein '
            '(e.g. "P0DTC2"). Use null if unknown.\n'
            "No markdown, no comments, no extra text."
        )
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip any accidental markdown fences
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(raw)
        location = result.get("location") or virus_name
        pdb_id   = result.get("pdb_id") or None
        logger.info("Groq returned location='%s', pdb_id='%s' for %s", location, pdb_id, virus_name)
        return {"location": location, "pdb_id": pdb_id}
    except Exception as exc:
        logger.warning("Groq virus-info fetch failed for '%s': %s", virus_name, exc)
        return {"location": virus_name, "pdb_id": None}


def fetch_coordinates_from_groq(location_text: str) -> Optional[tuple[float, float]]:
    """
    Last-resort geocoding: ask Groq to return lat/lng directly.
    Used when Nominatim cannot parse the location string.
    """
    client = _groq_client()
    if not client:
        return None
    try:
        prompt = (
            f"What are the latitude and longitude coordinates of '{location_text}'?\n"
            "Reply ONLY in valid JSON with keys \"lat\" and \"lng\" as floats.\n"
            "Example: {\"lat\": 31.5204, \"lng\": 74.3587}\n"
            "No markdown, no comments."
        )
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        coords = json.loads(raw)
        lat = float(coords["lat"])
        lng = float(coords["lng"])
        logger.info("Groq geocode: '%s' → (%.4f, %.4f)", location_text, lat, lng)
        return lat, lng
    except Exception as exc:
        logger.warning("Groq geocode fallback failed for '%s': %s", location_text, exc)
        return None


# ---------------------------------------------------------------------------
# LRU result cache
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
            del self[oldest]

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value


result_cache:    _LRUCache     = _LRUCache(maxsize=RESULT_CACHE_MAXSIZE)
last_fetch_time: Optional[str] = None
chroma_col                     = None

# ---------------------------------------------------------------------------
# Geocoder (with Groq fallback)
# ---------------------------------------------------------------------------
geolocator = Nominatim(user_agent="proteinwatch_app", timeout=5)


def _clean_location_for_geocoding(raw: str) -> str:
    """
    Remove common non-geocodable suffixes that Groq sometimes includes.
    E.g. "Ebola_virus_Sudan" → "Sudan"
         "Ebola: Zaire"       → "Zaire"
    """
    # Replace underscores with spaces
    clean = raw.replace("_", " ").replace(":", ",").strip()
    # If it looks like a virus name (contains "virus", "strain", etc.), extract last word/phrase
    lower = clean.lower()
    for keyword in ("virus", "strain", "variant", "subtype", "serotype"):
        if keyword in lower:
            # Keep only the part after the last keyword occurrence
            idx = lower.rfind(keyword)
            remainder = clean[idx + len(keyword):].strip(" ,")
            if remainder:
                clean = remainder
            break
    return clean.strip()


@lru_cache(maxsize=GEO_CACHE_MAXSIZE)
def _geocode_cached(location_text: str) -> tuple[float, float]:
    """
    Three-step geocoding:
      1. Try Nominatim with the original string.
      2. Try Nominatim with a cleaned string (removes virus name artifacts).
      3. Ask Groq for lat/lng directly.
      4. Return Bahawalpur only if all three fail.
    """
    if not location_text or location_text.lower() in ("unknown", ""):
        return DEFAULT_LAT, DEFAULT_LNG

    # Step 1: try as-is
    try:
        result = geolocator.geocode(location_text)
        if result:
            logger.debug("Nominatim hit (raw): '%s'", location_text)
            return result.latitude, result.longitude
    except Exception as exc:
        logger.warning("Nominatim error (raw) for '%s': %s", location_text, exc)

    # Step 2: try cleaned string
    cleaned = _clean_location_for_geocoding(location_text)
    if cleaned and cleaned.lower() != location_text.lower():
        try:
            result = geolocator.geocode(cleaned)
            if result:
                logger.info(
                    "Nominatim hit (cleaned): '%s' → '%s'", location_text, cleaned
                )
                return result.latitude, result.longitude
        except Exception as exc:
            logger.warning("Nominatim error (cleaned) for '%s': %s", cleaned, exc)

    # Step 3: Groq lat/lng
    coords = fetch_coordinates_from_groq(location_text)
    if coords:
        return coords

    # Step 4: genuine fallback (Bahawalpur)
    logger.warning(
        "All geocoding attempts failed for '%s'. Using default coordinates.", location_text
    )
    return DEFAULT_LAT, DEFAULT_LNG


def get_lat_lng(location_text: str) -> dict:
    lat, lng = _geocode_cached(location_text)
    return {"lat": lat, "lng": lng}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ProteinWatch API",
    description="Intelligent Viral Biosurveillance Pipeline",
    version="2.1.0",
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

    client   = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_col = client.get_or_create_collection("viral_sequences")

    scheduler.start()
    update_schedule(AUTO_FETCH_INTERVAL)
    logger.info("🚀 ProteinWatch started. ChromaDB: %d sequences.", chroma_col.count())


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
    return {
        "status":       "ok",
        "db_count":     chroma_col.count() if chroma_col else 0,
        "result_cache": {"size": len(result_cache), "maxsize": RESULT_CACHE_MAXSIZE},
        "esm2_cache":   esm2_cache_info(),
    }


@app.post("/analyze", tags=["Analysis"])
async def analyze(body: dict):
    sequence: str = body.get("sequence", "").strip()

    if len(sequence) < 50:
        return JSONResponse({"error": "Sequence too short (minimum 50 residues)."}, 400)

    seq_hash = hashlib.md5(sequence.encode()).hexdigest()
    if seq_hash in result_cache:
        return result_cache[seq_hash]

    # --- Layer 1 + 2 in parallel ---
    kmer_result, esm2_result = await asyncio.gather(
        asyncio.to_thread(compute_novelty, sequence),
        asyncio.to_thread(danger_score, sequence),
    )

    matched_virus: str = kmer_result.get("closest_match", "Unknown")

    # --- Layer 3: Dynamic AI Fallback (Location & UniProt/PDB) ---
    pdb_id      = None
    ai_location = matched_virus  # will be overwritten by Groq

    if matched_virus in VIRUS_METADATA:
        pdb_id      = VIRUS_METADATA[matched_virus]["pdb_id"]
        ai_location = matched_virus  # known viruses keep their name for geocoding
    elif matched_virus and matched_virus.lower() != "unknown":
        dynamic_info = await asyncio.to_thread(fetch_dynamic_virus_info, matched_virus)
        pdb_id       = dynamic_info["pdb_id"]
        ai_location  = dynamic_info["location"]   # clean geocodable string from Groq

    # --- Geocoding (3-step with Groq fallback) ---
    # Allow caller to override location; otherwise use Groq-provided location
    final_location = body.get("location_text") or ai_location
    coords = await asyncio.to_thread(get_lat_lng, final_location)

    # --- Layer 4: Structure Fetching + Real pLDDT Scoring ---
    struct_path  = None
    struct_score = 0.0
    pdb_label    = pdb_id  # what we report back to the frontend

    if pdb_id:
        struct_path = await asyncio.to_thread(fetch_alphafold_structure, pdb_id)
        if struct_path:
            # Real pLDDT from AlphaFold B-factor column
            struct_score = await asyncio.to_thread(compute_structural_score, struct_path)
            logger.info("AlphaFold pLDDT score for %s: %.1f", pdb_id, struct_score)

    if not struct_path:
        # Novel sequence → fold on-the-fly with ESMFold
        logger.info("Novel sequence detected. Folding dynamically using ESMFold...")
        struct_path = await asyncio.to_thread(fold_with_esmfold, sequence)
        if struct_path:
            pdb_label    = "CUSTOM_FOLD"
            # Real pLDDT from ESMFold output (stored in B-factor column too)
            struct_score = await asyncio.to_thread(compute_structural_score, struct_path)
            logger.info("ESMFold pLDDT score: %.1f", struct_score)

    # If structure entirely unavailable, use ESM-2 score as proxy
    if struct_score == 0.0:
        struct_score = esm2_result.get("danger_score", 50.0)
        logger.info(
            "No structure available; using ESM-2 score (%.1f) as structural proxy.",
            struct_score,
        )

    # --- Threat index ---
    kmer_score:  float = kmer_result.get("novelty_score", 50.0)
    esm2_score:  float = esm2_result.get("danger_score",  50.0)
    threat_index: float = round(
        kmer_score   * 0.25
        + esm2_score   * 0.45
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
        "pdb_id":           pdb_label,
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
    scores = {"combined": threat_index, "kmer": kmer, "esm2": esm2}

    async def _generator():
        try:
            for chunk in generate_brief_streaming(sequence, scores):
                yield f"data: {chunk}\n\n"
        except Exception as exc:
            yield f"data: Error generating brief: {exc}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generator(), media_type="text/event-stream")


@app.post("/simulate-action", tags=["Actions"])
async def simulate_action(body: dict):
    return simulate_alert_dispatch(
        body.get("sequence_id",  "unknown"),
        body.get("threat_index", 75.0),
        body.get("virus_name",   "Unknown Virus"),
    )


@app.post("/scheduler/update", tags=["System"])
async def update_sched(body: dict):
    return update_schedule(body.get("label", "6h"))


@app.get("/history", tags=["Analysis"])
async def history(limit: int = 50):
    sorted_results = sorted(
        result_cache.values(),
        key=lambda x: x.get("threat_index", 0),
        reverse=True,
    )
    return sorted_results[:limit]


@app.get("/structure/{uniprot_id}", tags=["Structure"])
async def get_structure(uniprot_id: str):
    # CUSTOM_FOLD is always in tmp/
    if uniprot_id == "CUSTOM_FOLD":
        path = "tmp/novel_query.pdb"
    else:
        path = f"data/structures/{uniprot_id}.pdb"

    if os.path.exists(path):
        with open(path) as fh:
            return {"pdb": fh.read(), "source": uniprot_id}

    return JSONResponse({"error": f"Structure '{uniprot_id}' not found."}, 404)


@app.get("/agent-trace/{analysis_id}", tags=["Actions"])
async def get_trace(analysis_id: str):
    for result in result_cache.values():
        if result.get("analysis_id") == analysis_id:
            alert = result.get("alert") or {}
            return alert.get("agent_trace", [])
    return []


@app.exception_handler(Exception)
async def _global_error_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url, exc)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "fallback": "Using cached data"},
    )