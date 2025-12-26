"""
Remove vector index for exact search.
For 28K rows, exact search is fast enough and 100% accurate.

Run: python fix_index.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
import time

PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def main():
    print("=" * 60)
    print("FIX: REMOVE INDEX FOR EXACT SEARCH")
    print("=" * 60)
    
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    # Check current state
    print("\n[1] Current state...")
    cur.execute("SELECT COUNT(*) FROM chunks")
    count = cur.fetchone()[0]
    print(f"    Chunks: {count:,}")
    
    # Check existing indexes
    cur.execute("""
        SELECT indexname FROM pg_indexes 
        WHERE tablename = 'chunks' AND indexname LIKE '%embedding%'
    """)
    indexes = cur.fetchall()
    print(f"    Vector indexes: {[i[0] for i in indexes]}")
    
    # Drop the IVFFlat index
    print("\n[2] Dropping IVFFlat index...")
    cur.execute("DROP INDEX IF EXISTS idx_chunks_embedding")
    conn.commit()
    print("    Done")
    
    # Test exact search speed
    print("\n[3] Testing exact search speed...")
    
    # Get a sample embedding for testing
    cur.execute("SELECT embedding FROM chunks WHERE id = 2")
    sample_embedding = cur.fetchone()[0]
    
    # Time the search
    start = time.time()
    cur.execute("""
        SELECT id, source, question, 
               1 - (embedding <=> %s::vector) as similarity
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT 10
    """, (sample_embedding, sample_embedding))
    results = cur.fetchall()
    elapsed = time.time() - start
    
    print(f"    Search time: {elapsed*1000:.1f}ms")
    print(f"    Top 5 results:")
    for row in results[:5]:
        print(f"      ID {row[0]} ({row[1]}): sim={row[3]:.4f}")
        print(f"        Q: {row[2][:50]}...")
    
    # Verify ID 2 is now at top
    if results[0][0] == 2:
        print("\n    ✓ ID 2 is now at top (exact match works!)")
    else:
        print(f"\n    ✗ ID 2 not at top, got ID {results[0][0]}")
    
    cur.close()
    conn.close()
    
    print("\n" + "=" * 60)
    print("EXACT SEARCH ENABLED")
    print("=" * 60)
    print(f"\nSearch time ~{elapsed*1000:.0f}ms is acceptable for 28K rows.")
    print("\nNow run: python diagnose_retrieval.py")


if __name__ == "__main__":
    main()
