"""
Generate Voyage AI embeddings and store chunks in Neon DB.
Run from project root: python backend/scripts/embed_and_store.py

Requires: chunks.json from chunk_books.py
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import voyageai
import psycopg2

load_dotenv()

# Voyage AI settings
EMBEDDING_MODEL = "voyage-3.5-lite"
BATCH_SIZE = 20


def load_chunks(chunks_file):
    """Load chunks from JSON file."""
    with open(chunks_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def embed_batch(client, texts):
    """Generate embeddings for a batch of texts."""
    result = client.embed(
        texts=texts,
        model=EMBEDDING_MODEL,
        input_type="document"  # CRITICAL: marks these as documents for retrieval
    )
    return result.embeddings


def store_chunks(conn, chunks_with_embeddings):
    """Store chunks and embeddings in Neon DB."""
    cur = conn.cursor()
    
    for chunk in chunks_with_embeddings:
        cur.execute("""
            INSERT INTO chunks (book_name, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s)
        """, (
            chunk['book_name'],
            chunk['chunk_index'],
            chunk['content'],
            chunk['embedding']
        ))
    
    conn.commit()
    cur.close()


def main():
    print("=" * 40)
    print("EMBED AND STORE CHUNKS")
    print("=" * 40)
    
    # Paths
    chunks_file = Path("data/processed/chunks.json")
    
    if not chunks_file.exists():
        print(f"\nERROR: {chunks_file} not found")
        print("Run chunk_books.py first!")
        return
    
    # Load chunks
    print(f"\n[1] Loading chunks from {chunks_file}...")
    chunks = load_chunks(chunks_file)
    print(f"    Loaded {len(chunks)} chunks")
    
    # Initialize Voyage client
    print("\n[2] Initializing Voyage AI client...")
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print("    ERROR: VOYAGE_API_KEY not found in .env")
        return
    
    client = voyageai.Client(api_key=api_key)
    print(f"    Model: {EMBEDDING_MODEL}")
    print(f"    input_type: document")
    
    # Connect to Neon
    print("\n[3] Connecting to Neon DB...")
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("    ERROR: DATABASE_URL not found in .env")
        return
    
    conn = psycopg2.connect(db_url)
    print("    Connected")
    
    # Clear existing data
    print("\n[4] Clearing existing chunks...")
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks")
    conn.commit()
    cur.close()
    print("    Cleared")
    
    # Process in batches
    print(f"\n[5] Generating embeddings and storing...")
    print(f"    Batch size: {BATCH_SIZE}")
    
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    total_stored = 0
    start_time = time.time()
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        
        # Get texts for embedding
        texts = [c['content'] for c in batch]
        
        # Generate embeddings
        embeddings = embed_batch(client, texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(batch, embeddings):
            chunk['embedding'] = embedding
        
        # Store in DB
        store_chunks(conn, batch)
        
        total_stored += len(batch)
        elapsed = time.time() - start_time
        rate = total_stored / elapsed if elapsed > 0 else 0
        
        print(f"    Batch {batch_num}/{total_batches}: {total_stored}/{len(chunks)} chunks ({rate:.1f}/sec)")
    
    conn.close()
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 40)
    print("DONE")
    print("=" * 40)
    print(f"\n  Chunks stored: {total_stored}")
    print(f"  Time: {elapsed:.1f} seconds")
    print(f"  Rate: {total_stored/elapsed:.1f} chunks/sec")


if __name__ == "__main__":
    main()
