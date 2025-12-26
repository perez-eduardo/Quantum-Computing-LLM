"""
Embed all Q&A pairs and store in Neon database.
Uses Voyage AI voyage-3.5-lite with input_type="document".

Run: python embed_qa.py
"""

import os
import csv
import io
import time
from pathlib import Path
from dotenv import load_dotenv
import voyageai
import psycopg2

# Load environment
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Settings
EMBEDDING_MODEL = "voyage-3.5-lite"
BATCH_SIZE = 20  # Voyage AI batch limit

# Data paths
DATA_DIR = PROJECT_ROOT / "data" / "raw"
CSV_FILES = [
    ("claude_qa_context.csv", "claude"),
    ("cot_qa_context.csv", "cot"),
    ("stackexchange_qa_context.csv", "stackexchange"),
]


def load_csv(filepath: Path) -> list:
    """Load Q&A pairs from CSV file."""
    rows = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        # Skip comment lines
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    
    reader = csv.DictReader(io.StringIO(''.join(lines)))
    
    for row in reader:
        question = row.get('question', '').strip()
        answer = row.get('answer', '').strip()
        
        if question and answer:
            rows.append({
                'question': question,
                'answer': answer
            })
    
    return rows


def embed_batch(client, texts: list) -> list:
    """Generate embeddings for a batch of texts."""
    result = client.embed(
        texts=texts,
        model=EMBEDDING_MODEL,
        input_type="document"  # Critical: marks as documents for retrieval
    )
    return result.embeddings


def format_for_embedding(question: str, answer: str) -> str:
    """Format Q&A pair for embedding."""
    return f"Q: {question} A: {answer}"


def main():
    print("=" * 60)
    print("EMBED Q&A PAIRS")
    print("=" * 60)
    
    # Check environment
    api_key = os.getenv("VOYAGE_API_KEY")
    db_url = os.getenv("DATABASE_URL")
    
    if not api_key:
        print("ERROR: VOYAGE_API_KEY not found")
        return
    if not db_url:
        print("ERROR: DATABASE_URL not found")
        return
    
    # Initialize clients
    print("\n[1] Initializing clients...")
    voyage = voyageai.Client(api_key=api_key)
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    print(f"    Voyage model: {EMBEDDING_MODEL}")
    print("    Database connected")
    
    # Check table is empty
    cur.execute("SELECT COUNT(*) FROM chunks")
    existing = cur.fetchone()[0]
    if existing > 0:
        print(f"\n    WARNING: Table has {existing:,} existing rows")
        response = input("    Clear and continue? (y/n): ")
        if response.lower() != 'y':
            print("    Aborted")
            return
        cur.execute("DELETE FROM chunks")
        conn.commit()
        print("    Cleared")
    
    # Load all CSV files
    print("\n[2] Loading CSV files...")
    all_rows = []
    
    for filename, source in CSV_FILES:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"    ERROR: {filepath} not found")
            continue
        
        rows = load_csv(filepath)
        for row in rows:
            row['source'] = source
        
        all_rows.extend(rows)
        print(f"    {filename}: {len(rows):,} rows")
    
    print(f"    Total: {len(all_rows):,} Q&A pairs")
    
    # Embed and store
    print(f"\n[3] Embedding and storing (batch_size={BATCH_SIZE})...")
    
    total_batches = (len(all_rows) + BATCH_SIZE - 1) // BATCH_SIZE
    stored = 0
    start_time = time.time()
    
    for i in range(0, len(all_rows), BATCH_SIZE):
        batch = all_rows[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        
        # Format texts for embedding
        texts = [format_for_embedding(r['question'], r['answer']) for r in batch]
        
        # Generate embeddings (with retry on rate limit)
        try:
            embeddings = embed_batch(voyage, texts)
        except Exception as e:
            print(f"\n    Error at batch {batch_num}: {e}")
            print("    Waiting 60s and retrying...")
            time.sleep(60)
            embeddings = embed_batch(voyage, texts)
        
        # Store in database
        for row, embedding in zip(batch, embeddings):
            cur.execute("""
                INSERT INTO chunks (source, question, answer, embedding)
                VALUES (%s, %s, %s, %s)
            """, (
                row['source'],
                row['question'],
                row['answer'],
                embedding
            ))
        
        conn.commit()
        stored += len(batch)
        
        # Progress
        elapsed = time.time() - start_time
        rate = stored / elapsed if elapsed > 0 else 0
        eta = (len(all_rows) - stored) / rate if rate > 0 else 0
        
        print(f"    Batch {batch_num}/{total_batches}: {stored:,}/{len(all_rows):,} "
              f"({rate:.1f}/sec, ETA: {eta/60:.1f}min)")
    
    cur.close()
    conn.close()
    
    # Summary
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("EMBEDDING COMPLETE")
    print("=" * 60)
    print(f"  Rows embedded: {stored:,}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Rate: {stored/elapsed:.1f} rows/sec")
    print("\nNext: python test_retrieval.py")


if __name__ == "__main__":
    main()
