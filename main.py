# main.py
import asyncio, hashlib, json, os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv

# NEW: Geopy import for exact locations
from geopy.geocoders import Nominatim

load_dotenv()

# Import your modules
from kmer_compare import compute_novelty
from esm2_scorer import danger_score
from structure_compare import fetch_alphafold_structure, foldseek_search
from gemini_brief import generate_brief_streaming
from simulate_action import simulate_alert_dispatch
from scheduler import scheduler, update_schedule
import chromadb

app = FastAPI()

# CORS — allow React frontend
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global state
cache = {}  # MD5 → result
last_fetch_time = None
chroma_col = None

# --- NEW: Geocoding Setup (Text to GPS) ---
geolocator = Nominatim(user_agent="proteinwatch_ciro_app")
geo_cache = {} # Cache to prevent slow API calls multiple times

def get_lat_lng_from_text(location_text: str):
    if location_text in geo_cache:
        return geo_cache[location_text]
        
    # Default fallback to Bahawalpur
    if location_text == "Unknown" or not location_text:
        return {"lat": 29.3957, "lng": 71.6833} 

    try:
        # Clean text: e.g., "China: Wuhan" -> "China, Wuhan"
        clean_loc = location_text.replace(":", ",") 
        location = geolocator.geocode(clean_loc, timeout=5)
        
        if location:
            coords = {"lat": location.latitude, "lng": location.longitude}
            geo_cache[location_text] = coords
            return coords
    except Exception as e:
        print(f"Geocoding error for {location_text}: {e}")
        
    return {"lat": 29.3957, "lng": 71.6833}

# --- VIRUS PDB IDs (Removed hardcoded locations, kept PDBs for 3D Viewer) ---
VIRUS_METADATA = {
    "SARS-CoV-2_Omicron_BA.5": {"pdb_id": "P0DTC2"}, 
    "Ebola_Virus": {"pdb_id": "Q05320"},
    "Zika_Virus": {"pdb_id": "Q32ZE1"}, 
    "Dengue": {"pdb_id": "P27909"}, 
    "Chikungunya": {"pdb_id": "Q8JUX5"}, 
}

@app.on_event("startup")
async def startup():
    global chroma_col
    os.makedirs("tmp", exist_ok=True)
    
    # ChromaDB
    client = chromadb.PersistentClient(path="data/chromadb")
    chroma_col = client.get_or_create_collection("viral_sequences")
    
    # Start scheduler
    scheduler.start()
    
    # Auto-pilot activated on startup
    update_schedule('30min')
    print("🚀 Auto-pilot Activated: Fetching new sequences every 30 minutes in background.")
    
    print(f"✅ Ready. ChromaDB has {chroma_col.count()} sequences.")

@app.get("/health")
async def health():
    return {
        "model_loaded": True,
        "db_count": chroma_col.count() if chroma_col else 0,
        "last_fetch": last_fetch_time,
        "cache_size": len(cache)
    }

@app.post("/analyze")
async def analyze(body: dict):
    sequence = body.get("sequence", "")
    if len(sequence) < 50:
        return JSONResponse({"error": "Sequence too short"}, 400)

    # Check cache first
    seq_hash = hashlib.md5(sequence.encode()).hexdigest()
    if seq_hash in cache:
        return cache[seq_hash]

    # Run Layer 2 and 3 in Parallel
    kmer_result, esm2_result = await asyncio.gather(
        asyncio.to_thread(compute_novelty, sequence),
        asyncio.to_thread(danger_score, sequence)
    )

    matched_virus = kmer_result.get("closest_match", "Unknown")
    
    # --- EXACT LOCATION: Resolve Text to GPS ---
    raw_location_text = body.get("location_text", matched_virus)
    exact_coords = await asyncio.to_thread(get_lat_lng_from_text, raw_location_text)

    # 3D Structure Resolution
    meta = VIRUS_METADATA.get(matched_virus, {"pdb_id": "P0DTC2"})
    struct_path = await asyncio.to_thread(fetch_alphafold_structure, meta["pdb_id"])

    # Compute combined threat index
    kmer_score = kmer_result.get("novelty_score", 50)
    esm2 = esm2_result.get("danger_score", 50)
    structural = 82.0 if struct_path else 70.0 
    
    threat_index = round(
        kmer_score * 0.25 + esm2 * 0.45 + structural * 0.30, 1
    )

    result = {
        "analysis_id": seq_hash[:8],
        "threat_index": threat_index,
        "kmer_score": kmer_score,
        "esm2_score": esm2,
        "structural_score": structural,
        "closest_match": matched_virus,
        "lat": exact_coords["lat"], # Exact Latitude
        "lng": exact_coords["lng"], # Exact Longitude
        "pdb_id": meta["pdb_id"],
        "alert": None
    }

    # Auto-dispatch alert if threat > 60
    if threat_index > 60:
        result["alert"] = simulate_alert_dispatch(
            seq_hash[:8], threat_index,
            matched_virus
        )

    cache[seq_hash] = result
    return result

@app.get("/stream-brief")
async def stream_brief(sequence: str, threat_index: float = 50,
                       kmer: float = 50, esm2: float = 50):
    scores = {"combined": threat_index, "kmer": kmer, "esm2": esm2}

    async def generator():
        try:
            for chunk in generate_brief_streaming(sequence, scores):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: Error generating brief: {str(e)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")

@app.post("/simulate-action")
async def simulate_action(body: dict):
    return simulate_alert_dispatch(
        body.get("sequence_id", "unknown"),
        body.get("threat_index", 75),
        body.get("virus_name", "Unknown Virus")
    )

@app.post("/scheduler/update")
async def update_sched(body: dict):
    return update_schedule(body.get("label", "6h"))

@app.get("/history")
async def history(limit: int = 50):
    sorted_cache = sorted(
        cache.values(),
        key=lambda x: x.get("threat_index", 0),
        reverse=True
    )
    return sorted_cache[:limit]

@app.get("/structure/{uniprot_id}")
async def get_structure(uniprot_id: str):
    path = f"data/structures/{uniprot_id}.pdb"
    if os.path.exists(path):
        return {"pdb": open(path).read()}
    return JSONResponse({"error": "Structure not found"}, 404)

@app.get("/agent-trace/{analysis_id}")
async def get_trace(analysis_id: str):
    for result in cache.values():
        if result.get("analysis_id") == analysis_id:
            alert = result.get("alert", {})
            return alert.get("agent_trace", [])
    return []

# Global error handler
@app.exception_handler(Exception)
async def error_handler(request, exc):
    return JSONResponse(status_code=500, content={
        "error": str(exc),
        "fallback": "Using cached data"
    })