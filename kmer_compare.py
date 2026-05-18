from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import DictVectorizer
from collections import Counter
import json
import numpy as np

# Load kmer_database.json
# Format: [{"virus_name": "X", "label": 1, "fingerprint": {"MEF": 1, ...}}, ...]
with open('data/kmer_database.json') as f:
    raw = json.load(f)

DB_NAMES = [entry['virus_name'] for entry in raw]
DB_VECS  = [entry['fingerprint'] for entry in raw]  # <-- the fix: extract fingerprint

print(f"[kmer] Loaded {len(DB_NAMES)} viruses. Example: {DB_NAMES[0]}")

# Fit vectorizer ONCE at startup
vec = DictVectorizer(sparse=False)
DB_MATRIX = vec.fit_transform(DB_VECS)

print(f"[kmer] Matrix shape: {DB_MATRIX.shape} — ready.")


def kmer_vector(seq: str, k: int = 3) -> dict:
    return dict(Counter([seq[i:i+k] for i in range(len(seq)-k+1)]))


def compute_novelty(new_seq: str) -> dict:
    try:
        new_kmer = kmer_vector(new_seq)
        new_vec  = vec.transform([new_kmer])
        sims     = cosine_similarity(new_vec, DB_MATRIX)[0]
        max_idx  = int(np.argmax(sims))

        return {
            'novelty_score':  round((1 - sims[max_idx]) * 100, 1),
            'closest_match':  DB_NAMES[max_idx],
            'similarity_pct': round(float(sims[max_idx]) * 100, 1),
            'top_5': [
                (DB_NAMES[i], round(float(sims[i]) * 100, 1))
                for i in np.argsort(sims)[-5:][::-1]
            ]
        }
    except Exception as e:
        return {
            'novelty_score': 50.0,
            'closest_match': 'unknown',
            'error': str(e)
        }