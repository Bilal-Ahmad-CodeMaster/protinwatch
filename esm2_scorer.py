# esm2_scorer.py
"""
ESM-2 Danger Score Module
--------------------------
Fine-tuned ESM-2 model served from Hugging Face Hub.
Uses lazy loading so the model is only pulled into memory
on the first actual inference request — not at import time.

FIXES APPLIED:
  - HF_TOKEN is now passed to from_pretrained() so private repos work.
  - MODEL_REPO default corrected to "arifhusnain/ProteinWatch" (was missing 'h').
  - Auth error is caught early with a clear message guiding the user.
"""

import os
import logging
from functools import lru_cache
from typing import Optional

import torch
from transformers import EsmTokenizer, EsmForSequenceClassification

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_REPO: str      = os.getenv("MODEL_REPO", "arifhusnain/ProteinWatch")
HF_TOKEN:   Optional[str] = os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN") or None
MAX_SEQ_LENGTH:       int = 1024
MIN_SEQ_LENGTH:       int = 50
INFERENCE_CACHE_SIZE: int = 128

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy model loader (singleton)
# ---------------------------------------------------------------------------
_tokenizer: Optional[EsmTokenizer]                  = None
_model:     Optional[EsmForSequenceClassification]  = None


def _load_model() -> tuple[EsmTokenizer, EsmForSequenceClassification]:
    """
    Download (or load from local HF cache) the ESM-2 model and tokenizer.
    Passes HF_TOKEN automatically so private/gated repos are accessible.
    Called once on first inference; subsequent calls reuse the globals.
    """
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    logger.info("Loading ESM-2 model from Hugging Face Hub: %s", MODEL_REPO)

    if not HF_TOKEN:
        logger.warning(
            "HF_API_TOKEN not set. Private repos will fail with 401. "
            "Set HF_API_TOKEN in your .env file."
        )

    try:
        _tokenizer = EsmTokenizer.from_pretrained(
            MODEL_REPO,
            token=HF_TOKEN,          # ← key fix: pass auth token
        )
        _model = EsmForSequenceClassification.from_pretrained(
            MODEL_REPO,
            token=HF_TOKEN,          # ← key fix: pass auth token
        )
        _model.eval()
        logger.info("✅ ESM-2 model loaded successfully from %s.", MODEL_REPO)

    except OSError as exc:
        # Repo not found or private without token
        if "401" in str(exc) or "not a valid model identifier" in str(exc):
            logger.critical(
                "❌ Cannot load model '%s'. "
                "Either the repo name is wrong OR it is private and HF_API_TOKEN is missing/invalid. "
                "Check MODEL_REPO and HF_API_TOKEN in your .env file.",
                MODEL_REPO,
            )
        else:
            logger.critical("❌ Failed to load ESM-2 model: %s", exc)
        raise RuntimeError(f"Could not load model '{MODEL_REPO}': {exc}") from exc

    except Exception as exc:
        logger.critical("❌ Failed to load ESM-2 model: %s", exc)
        raise RuntimeError(f"Could not load model '{MODEL_REPO}': {exc}") from exc

    return _tokenizer, _model


# ---------------------------------------------------------------------------
# Cached inference
# ---------------------------------------------------------------------------
@lru_cache(maxsize=INFERENCE_CACHE_SIZE)
def _cached_danger_score(sequence: str) -> dict:
    """
    Inner function wrapped by lru_cache.
    Deduplicates repeated sequences automatically.
    Evicts least-recently-used entry once INFERENCE_CACHE_SIZE is reached.
    """
    tokenizer, model = _load_model()

    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs  = torch.softmax(logits, dim=1)[0]
    danger = round(float(probs[1].item()) * 100, 1)
    safe   = round(float(probs[0].item()) * 100, 1)

    return {"danger_score": danger, "safe_prob": safe}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def danger_score(sequence: str) -> dict:
    """
    Compute the ESM-2 danger score for a viral protein sequence.

    Parameters
    ----------
    sequence : str
        Amino-acid sequence string (single-letter codes).

    Returns
    -------
    dict
        {
            "danger_score": float,   # 0–100, higher = more dangerous
            "safe_prob":    float,   # complement probability
        }
        On error, returns {"danger_score": 50.0, "error": "<message>"}.
    """
    if len(sequence) < MIN_SEQ_LENGTH:
        return {
            "danger_score": 0.0,
            "error": f"Sequence too short (minimum {MIN_SEQ_LENGTH} residues).",
        }

    try:
        return _cached_danger_score(sequence)
    except Exception as exc:
        logger.error("ESM-2 inference failed: %s", exc)
        return {"danger_score": 50.0, "error": str(exc)}


def cache_info() -> dict:
    """Expose lru_cache statistics (hits, misses, current size)."""
    info = _cached_danger_score.cache_info()
    return {
        "hits":     info.hits,
        "misses":   info.misses,
        "maxsize":  info.maxsize,
        "currsize": info.currsize,
    }


def clear_cache() -> None:
    """Flush the inference cache (useful for testing or memory pressure)."""
    _cached_danger_score.cache_clear()
    logger.info("ESM-2 inference cache cleared.")