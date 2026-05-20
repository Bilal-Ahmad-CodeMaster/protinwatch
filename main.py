# main.py
import asyncio
import hashlib
import logging
import os
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

# --- MODULE IMPORTS ---
from esm2_scorer import danger_score, cache_info as esm2_cache_info
from gemini_brief import generate_brief_streaming
from kmer_compare import compute_novelty
from scheduler import scheduler, update_schedule
from simulate_action import simulate_alert_dispatch
# FIX: Added compute_structural_score to imports
from structure_compare import fetch_alphafold_structure, foldseek_search, fold_with_esmfold, compute_structural_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("proteinwatch")

RESULT_CACHE_MAXSIZE = int(os.getenv("RESULT_CACHE_MAXSIZE", "100"))
GEO_CACHE_MAXSIZE    = int(os.getenv("GEO_CACHE_MAXSIZE",    "200"))
DEFAULT_LAT          = float(os.getenv("DEFAULT_LAT", "29.3957"))
DEFAULT_LNG          = float(os.getenv("DEFAULT_LNG", "71.6833"))
THREAT_ALERT_THRESHOLD = float(os.getenv("THREAT_ALERT_THRESHOLD", "60"))
AUTO_FETCH_INTERVAL  = os.getenv("AUTO_FETCH_INTERVAL", "30min")
CHROMA_PATH          = os.getenv("CHROMA_PATH", "data/chromadb")

VIRUS_METADATA = {
    "SARS-CoV-2_Omicron_BA.5": {"pdb_id": "P0DTC2"},
    "Ebola_Virus":              {"pdb_id": "Q05320"},
    "Zika_Virus":               {"pdb_id": "Q32ZE1"},
    "Dengue":                   {"pdb_id": "P27909"},
    "Chikungunya":              {"pdb_id": "Q8JUX5"},
}

def fetch_dynamic_virus_info(virus_name: str) -> dict:
    if virus_name == "Unknown":
        return {"location": "Unknown", "pdb_id": None}
    try:
        groq_key = os.environ.get('GROQ_API_KEY', '')
        if not groq_key: return {"location": virus_name, "pdb_id": None}

        client = Groq(api_key=groq_key)
        # FIX: Prompt made highly strict for Geocoding success
        prompt = f"""
        Provide information for the virus: '{virus_name}'.
        Reply ONLY in valid JSON format with exactly these two keys:
        1. "location": The specific City and Country ONLY (e.g. "Wuhan, China" or "Nzara, South Sudan"). Do not use broad regions.
        2. "pdb_id": The 4-letter PDB ID (e.g. "P0DTC2"). If unknown, use null.
        No other text.
        """
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        result = json.loads(response.choices[0].message.content)
        return {"location": result.get("location", virus_name), "pdb_id": result.get("pdb_id", None)}
    except Exception as e:
        logger.warning(f"Groq API fallback failed: {e}")
        return {"location": virus_name, "pdb_id": None}

class _LRUCache(OrderedDict):
    def __init__(self, maxsize=100):
        self._maxsize = maxsize
        super().__init__()
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self._maxsize:
            del self[next(iter(self))]
    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

result_cache = _LRUCache(maxsize=RESULT_CACHE_MAXSIZE)
last_fetch_time = None
chroma_col = None

geolocator = Nominatim(user_agent="proteinwatch_app", timeout=5)

@lru_cache(maxsize=GEO_CACHE_MAXSIZE)
def _geocode(location_text: str) -> tuple[float, float]:
    if not location_text or location_text.lower() == "unknown": return DEFAULT_LAT, DEFAULT_LNG
    try:
        result = geolocator.geocode(location_text)
        if result: return result.latitude, result.longitude
    except Exception as exc:
        logger.warning("Geocoding error for '%s': %s", location_text, exc)
    return DEFAULT_LAT, DEFAULT_LNG

def get_lat_lng(location_text: str) -> dict:
    lat, lng = _geocode(location_text)
    return {"lat": lat, "lng": lng}

app = FastAPI(title="ProteinWatch API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def on_startup():
    global chroma_col
    os.makedirs("tmp", exist_ok=True)
    os.makedirs("data/structures", exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_col = client.get_or_create_collection("viral_sequences")
    scheduler.start()
    update_schedule(AUTO_FETCH_INTERVAL)

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "db_count": chroma_col.count() if chroma_col else 0, "esm2_cache": esm2_cache_info()}

@app.post("/analyze", tags=["Analysis"])
async def analyze(body: dict):
    sequence: str = body.get("sequence", "").strip()
    if len(sequence) < 50: return JSONResponse({"error": "Sequence too short (minimum 50 residues)."}, 400)
    
    seq_hash = hashlib.md5(sequence.encode()).hexdigest()
    if seq_hash in result_cache: return result_cache[seq_hash]

    kmer_result, esm2_result = await asyncio.gather(
        asyncio.to_thread(compute_novelty, sequence),
        asyncio.to_thread(danger_score, sequence),
    )
    matched_virus: str = kmer_result.get("closest_match", "Unknown")

    pdb_id = None
    ai_location = matched_virus
    
    if matched_virus in VIRUS_METADATA:
        pdb_id = VIRUS_METADATA[matched_virus]["pdb_id"]
    elif matched_virus != "Unknown":
        dynamic_info = await asyncio.to_thread(fetch_dynamic_virus_info, matched_virus)
        pdb_id = dynamic_info["pdb_id"]
        ai_location = dynamic_info["location"]

    final_location = body.get("location_text") or ai_location
    coords = await asyncio.to_thread(get_lat_lng, final_location)

    struct_path = None
    if pdb_id:
        struct_path = await asyncio.to_thread(fetch_alphafold_structure, pdb_id)
            
    if not struct_path:
        logger.info("Novel sequence detected. Folding dynamically using ESMFold...")
        struct_path = await asyncio.to_thread(fold_with_esmfold, sequence)
        if struct_path: pdb_id = "CUSTOM_FOLD"

    # FIX: Calculate ACTUAL structure score from the PDB file! No more hardcoded 82.0 or 95.0.
    struct_score = await asyncio.to_thread(compute_structural_score, struct_path)

    kmer_score: float = kmer_result.get("novelty_score", 50.0)
    esm2_score: float = esm2_result.get("danger_score",  50.0)
    threat_index: float = round(kmer_score * 0.25 + esm2_score * 0.45 + struct_score * 0.30, 1)

    result = {
        "analysis_id": seq_hash[:8], "threat_index": threat_index,
        "kmer_score": kmer_score, "esm2_score": esm2_score, "structural_score": struct_score,
        "closest_match": matched_virus, "lat": coords["lat"], "lng": coords["lng"],
        "pdb_id": pdb_id, "alert": None,
    }

    if threat_index > THREAT_ALERT_THRESHOLD:
        result["alert"] = simulate_alert_dispatch(seq_hash[:8], threat_index, matched_virus)

    result_cache[seq_hash] = result
    return result

@app.get("/stream-brief", tags=["Analysis"])
async def stream_brief(sequence: str, threat_index: float = 50.0, kmer: float = 50.0, esm2: float = 50.0):
    scores = {"combined": threat_index, "kmer": kmer, "esm2": esm2}
    async def _generator():
        try:
            for chunk in generate_brief_streaming(sequence, scores): yield f"data: {chunk}\n\n"
        except Exception as exc: yield f"data: Error generating brief: {exc}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(_generator(), media_type="text/event-stream")

@app.get("/history", tags=["Analysis"])
async def history(limit: int = 50):
    return sorted(result_cache.values(), key=lambda x: x.get("threat_index", 0), reverse=True)[:limit]

@app.get("/structure/{pdb_id}", tags=["Structure"])
async def get_structure(pdb_id: str):
    # FIX: If frontend asks for CUSTOM_FOLD, return the temporary ESMFold generated file
    if pdb_id == "CUSTOM_FOLD":
        path = "tmp/novel_query.pdb"
    else:
        path = f"data/structures/{pdb_id}.pdb"
        
    if os.path.exists(path):
        with open(path) as fh:
            return {"pdb": fh.read()}
    return JSONResponse({"error": f"Structure '{pdb_id}' not found."}, 404)

@app.get("/agent-trace/{analysis_id}", tags=["Actions"])
async def get_trace(analysis_id: str):
    for result in result_cache.values():
        if result.get("analysis_id") == analysis_id:
            alert = result.get("alert") or {}
            return alert.get("agent_trace", [])
    return []

@app.exception_handler(Exception)
async def _global_error_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"error": str(exc)})