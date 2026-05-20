# structure_compare.py
"""
Structure Comparison Module
----------------------------
Fetches AlphaFold structures, folds novel sequences with ESMFold,
and extracts REAL pLDDT confidence scores from B-factor columns.

FIXES APPLIED:
  - compute_structural_score() is the single public entry point for scoring.
  - No hardcoded scores anywhere (was 82.0 / 95.0).
  - ESMFold pLDDT is read from the PDB it returns (B-factor column, same as AlphaFold).
  - FoldSeek is gracefully skipped when not installed.
"""

import requests
import subprocess
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STRUCTURES_DIR = Path("data/structures/")
STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# AlphaFold DB fetcher
# ---------------------------------------------------------------------------
def fetch_alphafold_structure(uniprot_id: str) -> Optional[str]:
    """
    Fetch a structure from the AlphaFold EBI database by UniProt accession.

    Returns
    -------
    str or None
        Path to the saved PDB file, or None if unavailable.
    """
    local = STRUCTURES_DIR / f"{uniprot_id}.pdb"
    if local.exists():
        logger.debug("Structure found locally: %s", uniprot_id)
        return str(local)

    try:
        logger.info("Fetching AlphaFold structure for %s", uniprot_id)
        meta = requests.get(
            f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}",
            timeout=15,
        )
        if meta.status_code == 200:
            pdb_url  = meta.json()[0]["pdbUrl"]
            pdb_resp = requests.get(pdb_url, timeout=30)
            local.write_text(pdb_resp.text, encoding="utf-8")
            logger.info("✅ AlphaFold structure saved: %s", uniprot_id)
            return str(local)
        else:
            logger.warning(
                "AlphaFold DB returned %d for %s", meta.status_code, uniprot_id
            )
    except Exception as exc:
        logger.warning("AlphaFold DB failed for %s: %s", uniprot_id, exc)

    return None


# ---------------------------------------------------------------------------
# ESMFold (free, unlimited)
# ---------------------------------------------------------------------------
def fold_with_esmfold(sequence: str) -> Optional[str]:
    """
    Fold a novel sequence using the free ESM Atlas API.
    ESMFold stores per-residue confidence in B-factor (same format as AlphaFold),
    so extract_plddt_confidence() works on its output without modification.

    Returns
    -------
    str or None
        Path to the written PDB file, or None on failure.
    """
    try:
        logger.info("Running ESMFold for novel sequence (len=%d)", len(sequence))
        resp = requests.post(
            "https://api.esmatlas.com/foldSequence/v1/pdb/",
            data=sequence[:400],   # API limit
            timeout=60,
        )
        if resp.status_code == 200:
            os.makedirs("tmp", exist_ok=True)
            pdb_path = "tmp/novel_query.pdb"
            with open(pdb_path, "w", encoding="utf-8") as fh:
                fh.write(resp.text)
            logger.info("✅ ESMFold completed: %s", pdb_path)
            return pdb_path
        else:
            logger.warning("ESMFold API returned %d", resp.status_code)
    except Exception as exc:
        logger.error("ESMFold failed: %s", exc)

    return None


# ---------------------------------------------------------------------------
# pLDDT extraction
# ---------------------------------------------------------------------------
def extract_plddt_confidence(pdb_path: str) -> float:
    """
    Parse the B-factor column (columns 61-66) of every ATOM record in a PDB file.
    Both AlphaFold and ESMFold store per-residue pLDDT there.

    pLDDT ranges 0–100; higher = more confident structure prediction.

    Returns
    -------
    float
        Mean pLDDT across all ATOM records, or 0.0 on failure.
    """
    if not pdb_path or not os.path.exists(pdb_path):
        logger.warning("PDB file not found: %s", pdb_path)
        return 0.0

    try:
        scores = []
        with open(pdb_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("ATOM"):
                    try:
                        b = float(line[60:66].strip())
                        scores.append(b)
                    except (ValueError, IndexError):
                        continue

        if scores:
            avg = round(sum(scores) / len(scores), 1)
            logger.debug(
                "pLDDT from %s: mean=%.1f (n=%d atoms)", pdb_path, avg, len(scores)
            )
            return avg

        logger.warning("No B-factors (pLDDT) found in %s", pdb_path)
        return 0.0

    except Exception as exc:
        logger.error("Failed to extract pLDDT from %s: %s", pdb_path, exc)
        return 0.0


# ---------------------------------------------------------------------------
# Structural confidence score (public entry point)
# ---------------------------------------------------------------------------
def compute_structural_score(pdb_path: Optional[str]) -> float:
    """
    Return the structural confidence score for a PDB file.

    Uses pLDDT extracted from the B-factor column.
    Returns 0.0 (not a hardcoded placeholder) when the file is unavailable.

    Parameters
    ----------
    pdb_path : str or None

    Returns
    -------
    float  — pLDDT confidence (0–100)
    """
    if not pdb_path or not os.path.exists(pdb_path):
        logger.debug("No structure available for scoring.")
        return 0.0

    score = extract_plddt_confidence(pdb_path)
    logger.info("Structural confidence score: %.1f", score)
    return score


# ---------------------------------------------------------------------------
# FoldSeek (optional — skipped gracefully if not installed)
# ---------------------------------------------------------------------------
def foldseek_search(pdb_path: str) -> dict:
    """
    Search a PDB structure against a local FoldSeek database.
    Returns TM-score and match info.
    Silently skips if FoldSeek is not installed.
    """
    if not pdb_path or not os.path.exists(pdb_path):
        logger.warning("PDB path invalid for FoldSeek: %s", pdb_path)
        return {"tm_score": 0.0, "match": "unknown", "error": "PDB not found"}

    try:
        logger.info("Running FoldSeek search for %s", pdb_path)
        subprocess.run(
            [
                "foldseek", "easy-search", pdb_path,
                "data/structureDB", "tmp/results.tsv", "tmp/",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )

        results_path = "tmp/results.tsv"
        if os.path.exists(results_path):
            with open(results_path) as fh:
                lines = fh.readlines()
            if lines:
                cols = lines[0].strip().split("\t")
                try:
                    tm    = float(cols[2]) if len(cols) > 2 else 0.0
                    match = cols[1]        if len(cols) > 1 else "unknown"
                    logger.info("FoldSeek: %s TM=%.3f", match, tm)
                    return {
                        "match":            match,
                        "tm_score":         round(tm * 100, 1),
                        "confirmed_threat": tm > 0.7,
                    }
                except (ValueError, IndexError) as exc:
                    logger.warning("FoldSeek result parse error: %s", exc)

    except FileNotFoundError:
        logger.info("FoldSeek not installed — skipping structural search.")
    except subprocess.CalledProcessError as exc:
        logger.error("FoldSeek subprocess error: %s", exc)
    except Exception as exc:
        logger.error("FoldSeek unexpected error: %s", exc)

    return {"tm_score": 0.0, "match": "no_match", "confirmed_threat": False}