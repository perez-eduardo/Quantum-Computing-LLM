"""
Create chunks table in Neon DB with pgvector.
Run from project root: python backend/scripts/create_tables.py
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Voyage 3.5-lite uses 1024 dimensions
EMBEDDING_DIM = 1024


def create_tables():
    """Create the chunks table with vector column."""
    print("=" * 40)
    print("CREATE TABLES")
    print("=" * 40)
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not found in .env")
        return False
    
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    # Drop existing table (fresh start)
    print("\n[1] Dropping existing table if exists...")
    cur.execute("DROP TABLE IF EXISTS chunks")
    
    # Create chunks table
    print("[2] Creating chunks table...")
    cur.execute(f"""
        CREATE TABLE chunks (
            id SERIAL PRIMARY KEY,
            book_name VARCHAR(100) NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding vector({EMBEDDING_DIM}),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create index for vector similarity search
    print("[3] Creating vector index...")
    cur.execute("""
        CREATE INDEX ON chunks 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 10)
    """)
    
    # Create index for book_name lookups
    print("[4] Creating book_name index...")
    cur.execute("CREATE INDEX ON chunks (book_name)")
    
    conn.commit()
    
    # Verify
    print("\n[5] Verifying...")
    cur.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'chunks'
    """)
    columns = cur.fetchall()
    
    print("  Columns:")
    for col_name, col_type in columns:
        print(f"    - {col_name}: {col_type}")
    
    cur.close()
    conn.close()
    
    print("\n" + "=" * 40)
    print("DONE: chunks table created")
    print("=" * 40)
    return True


if __name__ == "__main__":
    create_tables()
