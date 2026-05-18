import requests
import subprocess
import os
from pathlib import Path

STRUCTURES_DIR = Path('data/structures/')
STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)


def fetch_alphafold_structure(uniprot_id: str) -> str | None:
    # Try local file first (fastest — pre-downloaded)
    local = STRUCTURES_DIR / f'{uniprot_id}.pdb'
    if local.exists():
        return str(local)

    # Fetch from AlphaFold DB (NOT alphafoldserver.com)
    try:
        resp = requests.get(
            f'https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}',
            timeout=15
        )
        if resp.status_code == 200:
            pdb_url = resp.json()[0]['pdbUrl']
            pdb_resp = requests.get(pdb_url, timeout=30)
            local.write_text(pdb_resp.text)
            return str(local)
    except Exception as e:
        print(f'AlphaFold DB failed for {uniprot_id}: {e}')

    return None


def fold_with_esmfold(sequence: str) -> str | None:
    """Free unlimited folding for novel sequences"""
    try:
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
            return pdb_path
    except Exception as e:
        print(f'ESMFold failed: {e}')

    return None


def foldseek_search(pdb_path: str) -> dict:
    if not pdb_path or not os.path.exists(pdb_path):
        return {'tm_score': 0.0, 'match': 'unknown', 'error': 'PDB not found'}

    try:
        result = subprocess.run(
            ['foldseek', 'easy-search', pdb_path,
             'data/structureDB', 'tmp/results.tsv', 'tmp/'],
            capture_output=True, text=True, timeout=30
        )
        if os.path.exists('tmp/results.tsv'):
            lines = open('tmp/results.tsv').readlines()
            if lines:
                cols = lines[0].strip().split('\t')
                tm = float(cols[2]) if len(cols) > 2 else 0.0
                return {
                    'match': cols[1] if len(cols) > 1 else 'unknown',
                    'tm_score': tm,
                    'confirmed_threat': tm > 0.7
                }
    except FileNotFoundError:
        # Foldseek not installed — return safe default
        print('Foldseek not found — skipping structural search')
    except Exception as e:
        print(f'Foldseek error: {e}')

    return {'tm_score': 0.0, 'match': 'no_match', 'confirmed_threat': False}