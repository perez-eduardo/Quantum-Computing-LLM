"""
Diagnose RAG retrieval failures.
Run from project root: python diagnose_rag.py
"""

import os
from dotenv import load_dotenv
import psycopg2
import voyageai

load_dotenv()

EMBEDDING_MODEL = "voyage-3.5-lite"

def main():
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cur = conn.cursor()
    voyage = voyageai.Client(api_key=os.getenv('VOYAGE_API_KEY'))

    failed_terms = [
        'amplitude amplification',
        'surface code', 
        'CSS code',
        'commutator',
        'fidelity'
    ]

    print('='*60)
    print('PART 1: Do chunks with these exact terms exist?')
    print('='*60)

    chunks_with_terms = {}
    
    for term in failed_terms:
        cur.execute('''
            SELECT id, book_name, chunk_index, 
                   LEFT(content, 300) as preview
            FROM chunks 
            WHERE LOWER(content) LIKE %s
            LIMIT 3
        ''', (f'%{term.lower()}%',))
        
        results = cur.fetchall()
        chunks_with_terms[term] = results
        print(f'\n"{term}": {len(results)} chunks found')
        for r in results:
            print(f'  [id={r[0]}] {r[1]}:{r[2]}')
            print(f'    {r[3][:150]}...')

    print('\n' + '='*60)
    print('PART 2: What does semantic search return for these queries?')
    print('='*60)

    questions = [
        ("What is amplitude amplification?", "amplitude amplification"),
        ("What is the surface code?", "surface code"),
        ("What is a CSS code?", "CSS code"),
        ("What is the commutator?", "commutator"),
        ("What is fidelity?", "fidelity"),
    ]

    for question, term in questions:
        # Get query embedding
        result = voyage.embed(
            texts=[question],
            model=EMBEDDING_MODEL,
            input_type="query"
        )
        query_emb = result.embeddings[0]

        # Search
        cur.execute('''
            SELECT id, book_name, chunk_index,
                   LEFT(content, 200) as preview,
                   1 - (embedding <=> %s::vector) as similarity
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT 3
        ''', (query_emb, query_emb))

        results = cur.fetchall()
        
        print(f'\nQuery: "{question}"')
        print(f'Looking for: "{term}"')
        
        # Check if any result contains the term
        found_correct = False
        for r in results:
            contains_term = term.lower() in r[3].lower()
            marker = "<<< CORRECT" if contains_term else ""
            if contains_term:
                found_correct = True
            print(f'  [{r[0]}] sim={r[4]:.4f} | {r[1]}:{r[2]} {marker}')
            print(f'       {r[3][:100]}...')
        
        if not found_correct and chunks_with_terms.get(term):
            print(f'  !!! Term exists in DB but was NOT retrieved !!!')
            # Show the chunk that SHOULD have been retrieved
            correct_chunk = chunks_with_terms[term][0]
            print(f'  Correct chunk: id={correct_chunk[0]} ({correct_chunk[1]}:{correct_chunk[2]})')

    print('\n' + '='*60)
    print('PART 3: Compare similarity scores')
    print('='*60)

    # For one example, compare the similarity of correct vs retrieved chunk
    test_question = "What is amplitude amplification?"
    test_term = "amplitude amplification"
    
    result = voyage.embed(
        texts=[test_question],
        model=EMBEDDING_MODEL,
        input_type="query"
    )
    query_emb = result.embeddings[0]

    # Get similarity of TOP result
    cur.execute('''
        SELECT id, 1 - (embedding <=> %s::vector) as similarity
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT 1
    ''', (query_emb, query_emb))
    top_result = cur.fetchone()
    
    # Get similarity of CORRECT chunk (if it exists)
    cur.execute('''
        SELECT id, 1 - (embedding <=> %s::vector) as similarity
        FROM chunks
        WHERE LOWER(content) LIKE %s
        LIMIT 1
    ''', (query_emb, f'%{test_term.lower()}%'))
    correct_result = cur.fetchone()

    print(f'\nQuery: "{test_question}"')
    if top_result:
        print(f'  Top retrieved chunk:    id={top_result[0]}, sim={top_result[1]:.4f}')
    if correct_result:
        print(f'  Correct chunk:          id={correct_result[0]}, sim={correct_result[1]:.4f}')
        if top_result and correct_result:
            diff = top_result[1] - correct_result[1]
            print(f'  Similarity gap:         {diff:.4f}')

    cur.close()
    conn.close()

    print('\n' + '='*60)
    print('DIAGNOSIS COMPLETE')
    print('='*60)


if __name__ == "__main__":
    main()
