# esm2_scorer.py
import os
import platform
import logging
from functools import lru_cache
from typing import Optional

import torch
from transformers import EsmTokenizer, EsmForSequenceClassification

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# FIX: Fixed typo in default repo name
MODEL_REPO: str = os.getenv("MODEL_REPO", "arifhusnain/ProteinWatch")
# FIX: Explicitly get the HF token from env
HF_TOKEN: str = os.getenv("HF_API_TOKEN", "")

MAX_SEQ_LENGTH: int = 1024
MIN_SEQ_LENGTH: int = 50
INFERENCE_CACHE_SIZE: int = 128

logger = logging.getLogger(__name__)
_tokenizer: Optional[EsmTokenizer] = None
_model: Optional[EsmForSequenceClassification] = None

def _load_model() -> tuple[EsmTokenizer, EsmForSequenceClassification]:
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    logger.info("Loading ESM-2 model from Hugging Face Hub: %s", MODEL_REPO)

    try:
        # FIX: Pass token explicitly to allow private repo access
        _tokenizer = EsmTokenizer.from_pretrained(MODEL_REPO, token=HF_TOKEN)
        _model = EsmForSequenceClassification.from_pretrained(MODEL_REPO, token=HF_TOKEN)
        _model.eval()

        logger.info("✅ ESM-2 model loaded successfully.")
    except Exception as exc:
        logger.critical("❌ Failed to load ESM-2 model: %s", exc)
        raise RuntimeError(f"Could not load model '{MODEL_REPO}': {exc}") from exc

    return _tokenizer, _model

@lru_cache(maxsize=INFERENCE_CACHE_SIZE)
def _cached_danger_score(sequence: str) -> dict:
    tokenizer, model = _load_model()
    inputs = tokenizer(
        sequence, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    danger = round(float(probs[1].item()) * 100, 1)
    safe   = round(float(probs[0].item()) * 100, 1)
    return {"danger_score": danger, "safe_prob": safe}

def danger_score(sequence: str) -> dict:
    if len(sequence) < MIN_SEQ_LENGTH:
        return {"danger_score": 0.0, "error": f"Sequence too short (minimum {MIN_SEQ_LENGTH} residues)."}
    try:
        return _cached_danger_score(sequence)
    except Exception as exc:
        logger.error("ESM-2 inference failed: %s", exc)
        return {"danger_score": 50.0, "error": str(exc)}

def cache_info() -> dict:
    info = _cached_danger_score.cache_info()
    return {"hits": info.hits, "misses": info.misses, "maxsize": info.maxsize, "currsize": info.currsize}

def clear_cache() -> None:
    _cached_danger_score.cache_clear()
    logger.info("ESM-2 inference cache cleared.")
    