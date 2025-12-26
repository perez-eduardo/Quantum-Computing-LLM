"""
Diagnose why exact Q&A matches aren't ranking highest.
Investigates the retrieval ranking issue.

Run: python diagnose_retrieval.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import voyageai
import psycopg2

PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

EMBEDDING_MODEL = "voyage-3.5-lite"


def main():
    api_key = os.getenv("VOYAGE_API_KEY")
    db_url = os.getenv("DATABASE_URL")
    
    voyage = voyageai.Client(api_key=api_key)
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    test_question = "What is a qubit?"
    
    print("=" * 70)
    print("DIAGNOSE RETRIEVAL RANKING")
    print("=" * 70)
    print(f"\nQuery: {test_question}")
    
    # Step 1: Find exact matches in database
    print("\n[1] SEARCHING FOR EXACT MATCHES IN DATABASE")
    print("=" * 70)
    
    cur.execute("""
        SELECT id, source, question, LEFT(answer, 100) as answer_preview
        FROM chunks
        WHERE LOWER(question) LIKE %s
        LIMIT 10
    """, (f"%{test_question.lower()}%",))
    
    exact_matches = cur.fetchall()
    print(f"Found {len(exact_matches)} entries containing '{test_question}':")
    
    for row in exact_matches:
        print(f"\n  ID: {row[0]} | Source: {row[1]}")
        print(f"  Q: {row[2][:80]}...")
        print(f"  A: {row[3]}...")
    
    # Step 2: Get query embedding
    print("\n[2] GENERATING QUERY EMBEDDING")
    print("=" * 70)
    
    query_result = voyage.embed(
        texts=[test_question],
        model=EMBEDDING_MODEL,
        input_type="query"
    )
    query_embedding = query_result.embeddings[0]
    print(f"Query embedding generated (dim={len(query_embedding)})")
    
    # Step 3: Check similarity of exact matches vs top results
    print("\n[3] COMPARING SIMILARITIES")
    print("=" * 70)
    
    if exact_matches:
        exact_ids = [row[0] for row in exact_matches]
        
        # Get similarities for exact matches
        print("\nSimilarities for EXACT MATCHES:")
        for eid in exact_ids:
            cur.execute("""
                SELECT id, source, question, 
                       1 - (embedding <=> %s::vector) as similarity
                FROM chunks
                WHERE id = %s
            """, (query_embedding, eid))
            row = cur.fetchone()
            if row:
                print(f"  ID {row[0]} ({row[1]}): sim={row[3]:.4f}")
                print(f"    Q: {row[2][:60]}...")
    
    # Get top results by similarity
    print("\nTop 10 results by SEMANTIC SIMILARITY:")
    cur.execute("""
        SELECT id, source, question,
               1 - (embedding <=> %s::vector) as similarity
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT 10
    """, (query_embedding, query_embedding))
    
    top_results = cur.fetchall()
    for row in top_results:
        marker = " <-- EXACT MATCH" if row[0] in [e[0] for e in exact_matches] else ""
        print(f"  ID {row[0]} ({row[1]}): sim={row[3]:.4f}{marker}")
        print(f"    Q: {row[2][:60]}...")
    
    # Step 4: Check how Claude vs StackExchange entries were embedded
    print("\n[4] CHECKING EMBEDDING FORMAT BY SOURCE")
    print("=" * 70)
    
    cur.execute("""
        SELECT source, COUNT(*), AVG(LENGTH(question)), AVG(LENGTH(answer))
        FROM chunks
        GROUP BY source
    """)
    
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]:,} entries, avg_q_len={row[2]:.0f}, avg_a_len={row[3]:.0f}")
    
    # Step 5: Sample comparison - embed the exact question fresh
    print("\n[5] FRESH EMBEDDING TEST")
    print("=" * 70)
    
    # Get the stored question text for an exact match
    if exact_matches:
        exact_id = exact_matches[0][0]
        cur.execute("SELECT question, answer FROM chunks WHERE id = %s", (exact_id,))
        stored_q, stored_a = cur.fetchone()
        
        print(f"Stored question: {stored_q[:80]}...")
        
        # Embed as document (how it was stored)
        doc_text = f"Q: {stored_q} A: {stored_a}"
        doc_result = voyage.embed(
            texts=[doc_text],
            model=EMBEDDING_MODEL,
            input_type="document"
        )
        fresh_doc_embedding = doc_result.embeddings[0]
        
        # Calculate similarity between query and fresh document embedding
        import numpy as np
        query_arr = np.array(query_embedding)
        doc_arr = np.array(fresh_doc_embedding)
        
        # Cosine similarity
        cos_sim = np.dot(query_arr, doc_arr) / (np.linalg.norm(query_arr) * np.linalg.norm(doc_arr))
        print(f"Fresh embedding similarity: {cos_sim:.4f}")
        
        # Now check what similarity the stored embedding has
        cur.execute("""
            SELECT 1 - (embedding <=> %s::vector) as similarity
            FROM chunks WHERE id = %s
        """, (query_embedding, exact_id))
        stored_sim = cur.fetchone()[0]
        print(f"Stored embedding similarity: {stored_sim:.4f}")
        
        if abs(cos_sim - stored_sim) > 0.01:
            print("\n  WARNING: Stored embedding differs from fresh embedding!")
            print("  This suggests the embedding was done differently.")
    
    # Step 6: Check the actual stored content format
    print("\n[6] CHECKING STORED CONTENT FORMAT")
    print("=" * 70)
    
    print("\nSample CLAUDE entry:")
    cur.execute("""
        SELECT id, question, LEFT(answer, 200) 
        FROM chunks WHERE source = 'claude' LIMIT 1
    """)
    row = cur.fetchone()
    if row:
        print(f"  ID: {row[0]}")
        print(f"  Q: {row[1]}")
        print(f"  A: {row[2]}...")
    
    print("\nSample STACKEXCHANGE entry:")
    cur.execute("""
        SELECT id, question, LEFT(answer, 200) 
        FROM chunks WHERE source = 'stackexchange' LIMIT 1
    """)
    row = cur.fetchone()
    if row:
        print(f"  ID: {row[0]}")
        print(f"  Q: {row[1]}")
        print(f"  A: {row[2]}...")
    
    cur.close()
    conn.close()
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
