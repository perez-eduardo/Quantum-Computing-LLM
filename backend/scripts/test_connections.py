"""
Test Voyage AI and Neon DB connections.
Run from project root: python backend/scripts/test_connections.py
"""

import os
from dotenv import load_dotenv

load_dotenv()


def test_voyage():
    """Test Voyage AI API connection."""
    print("\n[1] Testing Voyage AI...")
    
    import voyageai
    
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print("  ERROR: VOYAGE_API_KEY not found in .env")
        return False
    
    client = voyageai.Client(api_key=api_key)
    
    # Test embedding
    result = client.embed(
        texts=["What is a qubit?"],
        model="voyage-3.5-lite"
    )
    
    dims = len(result.embeddings[0])
    print(f"  OK: Got embedding with {dims} dimensions")
    return True


def test_neon():
    """Test Neon DB connection."""
    print("\n[2] Testing Neon DB...")
    
    import psycopg2
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("  ERROR: DATABASE_URL not found in .env")
        return False
    
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    # Test pgvector
    cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
    result = cur.fetchone()
    
    if result:
        print("  OK: Connected, pgvector enabled")
    else:
        print("  ERROR: pgvector not enabled")
        return False
    
    cur.close()
    conn.close()
    return True


if __name__ == "__main__":
    print("=" * 40)
    print("CONNECTION TESTS")
    print("=" * 40)
    
    voyage_ok = test_voyage()
    neon_ok = test_neon()
    
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    print(f"  Voyage AI: {'OK' if voyage_ok else 'FAILED'}")
    print(f"  Neon DB:   {'OK' if neon_ok else 'FAILED'}")
    
    if voyage_ok and neon_ok:
        print("\n  Ready to build RAG pipeline!")
