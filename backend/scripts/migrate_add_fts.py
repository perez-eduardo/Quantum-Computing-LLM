"""
Database migration: Add full-text search support.
Adds tsvector column and GIN index for hybrid search.

Run from project root: python backend/scripts/migrate_add_fts.py
"""

import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()


def main():
    print("=" * 60)
    print("MIGRATION: Add Full-Text Search Support")
    print("=" * 60)

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not found")
        return

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    # Check current state
    print("\n[1] Checking current schema...")
    cur.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'chunks' AND column_name = 'content_tsv'
    """)
    if cur.fetchone():
        print("    content_tsv column already exists")
        response = input("    Re-run migration anyway? (y/n): ")
        if response.lower() != 'y':
            print("    Aborting.")
            return

    # Add tsvector column
    print("\n[2] Adding content_tsv column...")
    cur.execute("""
        ALTER TABLE chunks 
        ADD COLUMN IF NOT EXISTS content_tsv tsvector
    """)
    conn.commit()
    print("    Done")

    # Populate tsvector from content
    print("\n[3] Populating tsvector column...")
    cur.execute("""
        UPDATE chunks 
        SET content_tsv = to_tsvector('english', content)
        WHERE content_tsv IS NULL
    """)
    updated = cur.rowcount
    conn.commit()
    print(f"    Updated {updated:,} rows")

    # Create GIN index
    print("\n[4] Creating GIN index...")
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv 
        ON chunks USING GIN (content_tsv)
    """)
    conn.commit()
    print("    Done")

    # Create trigger for auto-update on insert
    print("\n[5] Creating trigger for auto-update...")
    cur.execute("""
        CREATE OR REPLACE FUNCTION chunks_tsv_trigger() 
        RETURNS trigger AS $$
        BEGIN
            NEW.content_tsv := to_tsvector('english', NEW.content);
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql;
    """)
    
    cur.execute("""
        DROP TRIGGER IF EXISTS chunks_tsv_update ON chunks;
    """)
    
    cur.execute("""
        CREATE TRIGGER chunks_tsv_update 
        BEFORE INSERT OR UPDATE ON chunks
        FOR EACH ROW EXECUTE FUNCTION chunks_tsv_trigger();
    """)
    conn.commit()
    print("    Done")

    # Verify
    print("\n[6] Verifying...")
    cur.execute("SELECT COUNT(*) FROM chunks WHERE content_tsv IS NOT NULL")
    count = cur.fetchone()[0]
    print(f"    Chunks with tsvector: {count:,}")

    # Test search
    cur.execute("""
        SELECT id, book_name, ts_rank(content_tsv, query) as rank
        FROM chunks, plainto_tsquery('english', 'amplitude amplification') query
        WHERE content_tsv @@ query
        ORDER BY rank DESC
        LIMIT 3
    """)
    results = cur.fetchall()
    print(f"    Test search 'amplitude amplification': {len(results)} results")
    for r in results:
        print(f"      id={r[0]}, source={r[1]}, rank={r[2]:.4f}")

    cur.close()
    conn.close()

    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
