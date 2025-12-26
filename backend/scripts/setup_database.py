"""
Setup Neon database for RAG.
Drops existing chunks table and creates fresh one with pgvector.

Run: python setup_database.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg2

# Load environment
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Voyage 3.5-lite uses 1024 dimensions
EMBEDDING_DIM = 1024


def main():
    print("=" * 60)
    print("SETUP DATABASE")
    print("=" * 60)
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not found in .env")
        return
    
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    # Step 1: Check current state
    print("\n[1] Checking current state...")
    cur.execute("""
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_name = 'chunks'
    """)
    exists = cur.fetchone()[0] > 0
    
    if exists:
        cur.execute("SELECT COUNT(*) FROM chunks")
        count = cur.fetchone()[0]
        print(f"    Existing chunks table: {count:,} rows")
    else:
        print("    No existing chunks table")
    
    # Step 2: Drop existing table
    print("\n[2] Dropping existing table...")
    cur.execute("DROP TABLE IF EXISTS chunks CASCADE")
    conn.commit()
    print("    Done")
    
    # Step 3: Ensure pgvector extension
    print("\n[3] Enabling pgvector extension...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    print("    Done")
    
    # Step 4: Create fresh table
    print(f"\n[4] Creating chunks table (embedding dim={EMBEDDING_DIM})...")
    cur.execute(f"""
        CREATE TABLE chunks (
            id SERIAL PRIMARY KEY,
            source VARCHAR(50) NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            embedding vector({EMBEDDING_DIM}),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    print("    Done")
    
    # Step 5: Create vector index
    print("\n[5] Creating vector index...")
    cur.execute("""
        CREATE INDEX idx_chunks_embedding 
        ON chunks USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)
    conn.commit()
    print("    Done")
    
    # Step 6: Verify
    print("\n[6] Verifying...")
    cur.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'chunks'
        ORDER BY ordinal_position
    """)
    columns = cur.fetchall()
    
    print("    Columns:")
    for col_name, col_type in columns:
        print(f"      - {col_name}: {col_type}")
    
    cur.close()
    conn.close()
    
    print("\n" + "=" * 60)
    print("DATABASE READY")
    print("=" * 60)
    print("\nNext: python embed_qa.py")


if __name__ == "__main__":
    main()
