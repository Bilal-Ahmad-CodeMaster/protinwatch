# structure_compare.py (FIXED)
"""
Structure Comparison Module — ACTUAL CONFIDENCE SCORING
-------------------------------------------------------
This MUST compute structural confidence scores dynamically.
Do NOT return hardcoded 82.0.
"""

import requests
import subprocess
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STRUCTURES_DIR = Path('data/structures/')
STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)


def fetch_alphafold_structure(uniprot_id: str) -> Optional[str]:
    """
    Fetch AlphaFold structure and extract pLDDT confidence score.
    
    Returns
    -------
    str or None
        Path to PDB file if successful, else None
    """
    # Try local file first (fastest — pre-downloaded)
    local = STRUCTURES_DIR / f'{uniprot_id}.pdb'
    if local.exists():
        logger.debug("Structure found locally: %s", uniprot_id)
        return str(local)

    # Fetch from AlphaFold DB (NOT alphafoldserver.com)
    try:
        logger.info("Fetching AlphaFold structure for %s", uniprot_id)
        resp = requests.get(
            f'https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}',
            timeout=15
        )
        if resp.status_code == 200:
            pdb_url = resp.json()[0]['pdbUrl']
            pdb_resp = requests.get(pdb_url, timeout=30)
            local.write_text(pdb_resp.text)
            logger.info("✅ AlphaFold structure saved: %s", uniprot_id)
            return str(local)
    except Exception as e:
        logger.warning('AlphaFold DB failed for %s: %s', uniprot_id, e)

    return None


def extract_plddt_confidence(pdb_path: str) -> float:
    """
    Extract pLDDT (predicted Local Distance Difference Test) score from PDB B-factor.
    
    AlphaFold stores confidence in the B-factor column (columns 61-66).
    pLDDT ranges 0-100; higher = more confident.
    
    Parameters
    ----------
    pdb_path : str
        Path to PDB file
    
    Returns
    -------
    float
        Average pLDDT confidence (0-100), or 0.0 if parsing fails
    """
    if not pdb_path or not os.path.exists(pdb_path):
        logger.warning("PDB file not found: %s", pdb_path)
        return 0.0

    try:
        with open(pdb_path, 'r') as f:
            pdb_content = f.read()
        
        plddt_scores = []
        for line in pdb_content.split('\n'):
            if line.startswith('ATOM'):
                try:
                    # PDB format: B-factor is in columns 61-66
                    b_factor = float(line[60:66].strip())
                    plddt_scores.append(b_factor)
                except (ValueError, IndexError):
                    continue
        
        if plddt_scores:
            avg_confidence = sum(plddt_scores) / len(plddt_scores)
            # Map 0-100 pLDDT to 0-100 confidence score
            confidence = round(avg_confidence, 1)
            logger.debug(
                "Extracted pLDDT from %s: avg=%.1f (from %d atoms)",
                pdb_path, confidence, len(plddt_scores)
            )
            return confidence
        else:
            logger.warning("No B-factors found in %s", pdb_path)
            return 0.0
            
    except Exception as e:
        logger.error("Failed to extract pLDDT from %s: %s", pdb_path, e)
        return 0.0


def fold_with_esmfold(sequence: str) -> Optional[str]:
    """
    Free unlimited folding for novel sequences using ESM Atlas.
    Returns path to PDB file.
    """
    try:
        logger.info("Running ESMFold for novel sequence (len=%d)", len(sequence))
        resp = requests.post(
            'https://api.esmatlas.com/foldSequence/v1/pdb/',
            data=sequence[:400],  # Truncate very long sequences
            timeout=60
        )
        if resp.status_code == 200:
            os.makedirs('tmp', exist_ok=True)
            pdb_path = 'tmp/novel_query.pdb'
            with open(pdb_path, 'w') as f:
                f.write(resp.text)
            logger.info("✅ ESMFold completed: %s", pdb_path)
            return pdb_path
    except Exception as e:
        logger.error('ESMFold failed: %s', e)

    return None


def foldseek_search(pdb_path: str) -> dict:
    """
    Search PDB structure against database using FoldSeek.
    Returns TM-score and match confidence.
    
    CRITICAL: Do NOT return hardcoded tm_score.
    """
    if not pdb_path or not os.path.exists(pdb_path):
        logger.warning("PDB path invalid: %s", pdb_path)
        return {'tm_score': 0.0, 'match': 'unknown', 'error': 'PDB not found'}

    try:
        logger.info("Running FoldSeek search for %s", pdb_path)
        result = subprocess.run(
            ['foldseek', 'easy-search', pdb_path,
             'data/structureDB', 'tmp/results.tsv', 'tmp/'],
            capture_output=True, text=True, timeout=30
        )
        
        if os.path.exists('tmp/results.tsv'):
            with open('tmp/results.tsv') as f:
                lines = f.readlines()
            
            if lines:
                cols = lines[0].strip().split('\t')
                try:
                    tm = float(cols[2]) if len(cols) > 2 else 0.0
                    match = cols[1] if len(cols) > 1 else 'unknown'
                    
                    logger.info(
                        "FoldSeek match: %s (TM=%.3f)",
                        match, tm
                    )
                    
                    return {
                        'match': match,
                        'tm_score': round(tm * 100, 1),  # Convert to 0-100 scale
                        'confirmed_threat': tm > 0.7
                    }
                except (ValueError, IndexError) as e:
                    logger.warning("Failed to parse FoldSeek result: %s", e)
                    
    except FileNotFoundError:
        logger.warning("FoldSeek not installed — skipping structural search")
    except Exception as e:
        logger.error('FoldSeek error: %s', e)

    return {'tm_score': 0.0, 'match': 'no_match', 'confirmed_threat': False}


def compute_structural_score(pdb_path: Optional[str]) -> float:
    """
    CRITICAL FIX: Compute actual structural confidence score.
    
    If PDB exists, extract pLDDT. Never return hardcoded 82.0.
    
    Parameters
    ----------
    pdb_path : str or None
        Path to PDB structure file
    
    Returns
    -------
    float
        Structural confidence score (0-100)
    """
    if not pdb_path or not os.path.exists(pdb_path):
        logger.debug("No structure available for scoring")
        return 0.0
    
    confidence = extract_plddt_confidence(pdb_path)
    logger.info("Structural confidence score: %.1f", confidence)
    return confidence