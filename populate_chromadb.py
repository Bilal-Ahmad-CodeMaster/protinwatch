"""
Run this ONCE before starting the server:
  python populate_chromadb.py
"""
import chromadb
import pandas as pd
import os

CSV_PATH = 'data/pandemic_training_data.csv'
CHROMA_PATH = 'data/chromadb'

client = chromadb.PersistentClient(path=CHROMA_PATH)
col = client.get_or_create_collection('viral_sequences')

# Skip if already populated
existing = col.count()
if existing > 0:
    print(f"Already populated: {existing} sequences. Skipping.")
    exit(0)

# Load CSV
df = pd.read_csv(CSV_PATH)
print(f"Loading {len(df)} sequences from {CSV_PATH}...")

# Required columns: check what we have
print(f"Columns: {list(df.columns)}")

# Batch add to ChromaDB (max 100 at a time)
BATCH = 100
added = 0

for i in range(0, len(df), BATCH):
    batch = df.iloc[i:i+BATCH]
    
    ids       = [f"seq_{i+j}" for j in range(len(batch))]
    documents = batch['sequence'].tolist() if 'sequence' in batch.columns else batch.iloc[:,0].tolist()
    
    # Build metadata from available columns
    metadatas = []
    for _, row in batch.iterrows():
        meta = {
            'virus_name': str(row.get('virus_name', row.get('name', f'virus_{i}'))),
            'label':      str(row.get('label', '0')),
            'notes':      str(row.get('notes', row.get('description', 'No notes')))
        }
        metadatas.append(meta)
    
    col.add(ids=ids, documents=documents, metadatas=metadatas)
    added += len(batch)
    print(f"  Added {added}/{len(df)}...")

print(f"\n✅ Done! ChromaDB now has {col.count()} sequences.")