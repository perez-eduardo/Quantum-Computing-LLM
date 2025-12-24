"""
Embed Q&A pairs and add to chunks table.
Run from project root: python backend/scripts/embed_qa_to_chunks.py

Adds Q&A pairs alongside existing book chunks.
Does NOT delete existing book chunks.
"""

import os
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import voyageai
import psycopg2

load_dotenv()

EMBEDDING_MODEL = "voyage-3.5-lite"
BATCH_SIZE = 20


def format_qa(row):
    """Format Q&A pair as single content string."""
    return f"Q: {row['question']}\nA: {row['answer']}"


def embed_batch(client, texts):
    """Generate embeddings for a batch of texts."""
    result = client.embed(
        texts=texts,
        model=EMBEDDING_MODEL,
        input_type="document"
    )
    return result.embeddings


def main():
    print("=" * 60)
    print("EMBED Q&A PAIRS TO CHUNKS TABLE")
    print("=" * 60)

    # Paths
    qa_file = Path("data/processed/combined_qa_v4_filtered.csv")

    if not qa_file.exists():
        print(f"\nERROR: {qa_file} not found")
        return

    # Load Q&A
    print(f"\n[1] Loading Q&A from {qa_file}...")
    df = pd.read_csv(qa_file)
    print(f"    Loaded {len(df):,} Q&A pairs")
    print(f"    Sources: {df['source'].value_counts().to_dict()}")

    # Initialize Voyage
    print("\n[2] Initializing Voyage AI...")
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print("    ERROR: VOYAGE_API_KEY not found")
        return

    client = voyageai.Client(api_key=api_key)
    print(f"    Model: {EMBEDDING_MODEL}")

    # Connect to Neon
    print("\n[3] Connecting to Neon DB...")
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("    ERROR: DATABASE_URL not found")
        return

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    print("    Connected")

    # Check existing chunks
    cur.execute("SELECT COUNT(*) FROM chunks")
    existing_count = cur.fetchone()[0]
    print(f"    Existing chunks: {existing_count:,}")

    # Check if Q&A already added
    cur.execute("SELECT COUNT(*) FROM chunks WHERE book_name IN ('claude_synthetic', 'stackexchange', 'cot_reasoning')")
    qa_count = cur.fetchone()[0]
    if qa_count > 0:
        print(f"\n    WARNING: {qa_count:,} Q&A chunks already exist!")
        response = input("    Delete existing Q&A chunks and re-embed? (y/n): ")
        if response.lower() == 'y':
            cur.execute("DELETE FROM chunks WHERE book_name IN ('claude_synthetic', 'stackexchange', 'cot_reasoning')")
            conn.commit()
            print(f"    Deleted {qa_count:,} existing Q&A chunks")
        else:
            print("    Aborting.")
            return

    # Format Q&A pairs
    print("\n[4] Formatting Q&A pairs...")
    df['content'] = df.apply(format_qa, axis=1)

    # Estimate tokens
    total_chars = df['content'].str.len().sum()
    est_tokens = total_chars // 4
    print(f"    Total characters: {total_chars:,}")
    print(f"    Estimated tokens: {est_tokens:,}")

    # Process in batches
    print(f"\n[5] Embedding and storing Q&A pairs...")
    print(f"    Batch size: {BATCH_SIZE}")

    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    total_stored = 0
    start_time = time.time()

    for i in range(0, len(df), BATCH_SIZE):
        batch_df = df.iloc[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        # Get content for embedding
        texts = batch_df['content'].tolist()

        # Generate embeddings
        try:
            embeddings = embed_batch(client, texts)
        except Exception as e:
            print(f"\n    ERROR at batch {batch_num}: {e}")
            print("    Waiting 60s and retrying...")
            time.sleep(60)
            embeddings = embed_batch(client, texts)

        # Store in DB
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            cur.execute("""
                INSERT INTO chunks (book_name, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
            """, (
                row['source'],
                idx + i,
                row['content'],
                embeddings[idx]
            ))

        conn.commit()
        total_stored += len(batch_df)

        elapsed = time.time() - start_time
        rate = total_stored / elapsed if elapsed > 0 else 0
        eta = (len(df) - total_stored) / rate if rate > 0 else 0

        print(f"    Batch {batch_num}/{total_batches}: {total_stored:,}/{len(df):,} ({rate:.1f}/sec, ETA: {eta/60:.1f}min)")

    cur.close()
    conn.close()

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Q&A pairs added: {total_stored:,}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Rate: {total_stored/elapsed:.1f} pairs/sec")

    print("\n  Next: Re-run retrieval tests")
    print("  python backend/scripts/test_retrieval_quality.py")


if __name__ == "__main__":
    main()
