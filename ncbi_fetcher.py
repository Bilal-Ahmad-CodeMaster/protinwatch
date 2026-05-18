# ncbi_fetcher.py
import requests
import time
import hashlib
import json
from typing import List, Dict

BASE = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'

def fetch_sequences(hours_back: int = 6) -> List[Dict]:
    try:
        search_resp = requests.get(BASE + 'esearch.fcgi', params={
            'db': 'protein',
            'retmax': 20,
            'retmode': 'json',
            'term': 'spike[Protein Name] AND Viruses[Organism]'
        }, timeout=10)
        search_resp.raise_for_status()
        ids = search_resp.json()['esearchresult']['idlist']
    except Exception as e:
        print(f'NCBI search failed: {e}. Using cached sequences.')
        return load_cached_sequences()

    sequences = []
    for ncbi_id in ids:
        try:
            from Bio import SeqIO
            from io import StringIO
            # Fetch as GenBank ('gb') instead of fasta to get exact country metadata
            resp = requests.get(BASE + 'efetch.fcgi', params={
                'db': 'protein', 'id': ncbi_id,
                'rettype': 'gb', 'retmode': 'text' 
            }, timeout=10)
            
            for rec in SeqIO.parse(StringIO(resp.text), 'genbank'):
                seq = str(rec.seq)
                
                # Extract Exact Country/Location Text
                exact_location = "Unknown"
                for feature in rec.features:
                    if feature.type == "source":
                        if 'country' in feature.qualifiers:
                            exact_location = feature.qualifiers['country'][0]
                            break
                        elif 'isolation_source' in feature.qualifiers:
                            exact_location = feature.qualifiers['isolation_source'][0]
                            break

                if len(seq) >= 200:
                    sequences.append({
                        'id': rec.id,
                        'sequence': seq,
                        'description': rec.description,
                        'location_text': exact_location, # Added to dictionary
                        'hash': hashlib.md5(seq.encode()).hexdigest()
                    })
        except Exception as e:
            print(f'Skipping {ncbi_id}: {e}')
        time.sleep(0.5)  # CRITICAL: respect NCBI rate limit

    return sequences if sequences else load_cached_sequences()

def load_cached_sequences() -> List[Dict]:
    try:
        with open('data/cached_sequences.json') as f:
            return json.load(f)
    except Exception:
        # Return a minimal fallback so the server never crashes
        return [{
            'id': 'fallback_001',
            'sequence': 'MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFR',
            'description': 'Fallback sequence',
            'location_text': 'China: Wuhan',
            'hash': 'fallback'
        }]