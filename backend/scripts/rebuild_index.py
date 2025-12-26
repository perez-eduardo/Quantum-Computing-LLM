"""
Rebuild the vector index after data is loaded.
IVFFlat must be built on populated data to work correctly.

Run: python rebuild_index.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg2

PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def main():
    print("=" * 60)
    print("REBUILD VECTOR INDEX")
    print("=" * 60)
    
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    # Check current state
    print("\n[1] Checking current state...")
    cur.execute("SELECT COUNT(*) FROM chunks")
    count = cur.fetchone()[0]
    print(f"    Chunks in table: {count:,}")
    
    if count == 0:
        print("    ERROR: No data in table. Run embed_qa.py first.")
        return
    
    # Drop existing index
    print("\n[2] Dropping existing index...")
    cur.execute("DROP INDEX IF EXISTS idx_chunks_embedding")
    conn.commit()
    print("    Done")
    
    # Calculate optimal list count
    # Rule of thumb: lists = sqrt(n) for n < 1M rows
    import math
    optimal_lists = max(10, int(math.sqrt(count)))
    print(f"\n[3] Calculating optimal parameters...")
    print(f"    Rows: {count:,}")
    print(f"    Optimal lists: {optimal_lists}")
    
    # Rebuild index
    print(f"\n[4] Rebuilding IVFFlat index (lists={optimal_lists})...")
    print("    This may take a minute...")
    
    cur.execute(f"""
        CREATE INDEX idx_chunks_embedding 
        ON chunks USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = {optimal_lists})
    """)
    conn.commit()
    print("    Done")
    
    # Test search
    print("\n[5] Testing search...")
    
    # Get a sample embedding
    cur.execute("SELECT embedding FROM chunks WHERE id = 2")
    sample_embedding = cur.fetchone()[0]
    
    # Search with new index
    cur.execute("""
        SELECT id, source, question, 
               1 - (embedding <=> %s::vector) as similarity
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT 5
    """, (sample_embedding, sample_embedding))
    
    results = cur.fetchall()
    print("    Top 5 results for ID 2's embedding:")
    for row in results:
        print(f"      ID {row[0]} ({row[1]}): sim={row[3]:.4f}")
        print(f"        Q: {row[2][:50]}...")
    
    cur.close()
    conn.close()
    
    print("\n" + "=" * 60)
    print("INDEX REBUILT")
    print("=" * 60)
    print("\nNow run: python test_retrieval.py")


if __name__ == "__main__":
    main()
