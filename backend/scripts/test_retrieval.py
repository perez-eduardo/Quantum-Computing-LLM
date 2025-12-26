"""
Test retrieval quality after embedding.
Verifies that semantic search returns relevant Q&A pairs.

Run: python test_retrieval.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import voyageai
import psycopg2

# Load environment
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

EMBEDDING_MODEL = "voyage-3.5-lite"

# Test cases: question -> expected keywords in retrieved content
TEST_CASES = [
    ("What is a qubit?", ["qubit", "quantum bit", "superposition", "state"]),
    ("What is superposition?", ["superposition", "multiple states", "simultaneously"]),
    ("What is quantum entanglement?", ["entangle", "correlat", "particle"]),
    ("What is a quantum gate?", ["gate", "operation", "unitary", "qubit"]),
    ("What is Shor's algorithm?", ["shor", "factor", "prime", "integer"]),
    ("What is Grover's algorithm?", ["grover", "search", "speedup"]),
    ("What is quantum error correction?", ["error", "correct", "code", "logical"]),
    ("What is the Hadamard gate?", ["hadamard", "superposition", "H gate"]),
    ("What is decoherence?", ["decoherence", "environment", "noise"]),
    ("What is a quantum circuit?", ["circuit", "gate", "qubit"]),
]


class TestRetriever:
    def __init__(self):
        self.api_key = os.getenv("VOYAGE_API_KEY")
        self.db_url = os.getenv("DATABASE_URL")
        self.voyage = voyageai.Client(api_key=self.api_key)
    
    def search(self, query: str, top_k: int = 3) -> list:
        """Search for similar Q&A pairs."""
        # Embed query
        result = self.voyage.embed(
            texts=[query],
            model=EMBEDDING_MODEL,
            input_type="query"  # Critical: marks as query
        )
        query_embedding = result.embeddings[0]
        
        # Search
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT 
                source,
                question,
                answer,
                1 - (embedding <=> %s::vector) as similarity
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, top_k))
        
        results = []
        for row in cur.fetchall():
            results.append({
                'source': row[0],
                'question': row[1],
                'answer': row[2],
                'similarity': float(row[3])
            })
        
        cur.close()
        conn.close()
        
        return results


def check_relevance(results: list, keywords: list) -> tuple:
    """Check if any result contains expected keywords."""
    found = []
    
    for r in results:
        text = (r['question'] + ' ' + r['answer']).lower()
        for kw in keywords:
            if kw.lower() in text and kw not in found:
                found.append(kw)
    
    missing = [kw for kw in keywords if kw.lower() not in [f.lower() for f in found]]
    return found, missing


def main():
    print("=" * 60)
    print("TEST RETRIEVAL QUALITY")
    print("=" * 60)
    
    # Check database stats
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM chunks")
    total = cur.fetchone()[0]
    
    cur.execute("SELECT source, COUNT(*) FROM chunks GROUP BY source")
    by_source = {row[0]: row[1] for row in cur.fetchall()}
    
    cur.close()
    conn.close()
    
    print(f"\nDatabase stats:")
    print(f"  Total chunks: {total:,}")
    for source, count in by_source.items():
        print(f"    {source}: {count:,}")
    
    if total == 0:
        print("\nERROR: No chunks in database. Run embed_qa.py first.")
        return
    
    # Run tests
    print(f"\n{'='*60}")
    print(f"Running {len(TEST_CASES)} tests...")
    print("=" * 60)
    
    retriever = TestRetriever()
    
    passed = 0
    failed = 0
    failures = []
    
    for question, keywords in TEST_CASES:
        results = retriever.search(question, top_k=3)
        found, missing = check_relevance(results, keywords)
        
        # Pass if at least half of keywords found
        score = len(found) / len(keywords)
        status = "PASS" if score >= 0.5 else "FAIL"
        
        if status == "PASS":
            passed += 1
        else:
            failed += 1
            failures.append({
                'question': question,
                'keywords': keywords,
                'found': found,
                'missing': missing,
                'results': results
            })
        
        print(f"\n{status}: {question}")
        print(f"  Keywords: {found} ({score:.0%})")
        print(f"  Top result (sim={results[0]['similarity']:.4f}):")
        print(f"    Q: {results[0]['question'][:80]}...")
        print(f"    A: {results[0]['answer'][:100]}...")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{len(TEST_CASES)} ({100*passed/len(TEST_CASES):.0f}%)")
    print(f"Failed: {failed}/{len(TEST_CASES)}")
    
    # Show failures in detail
    if failures:
        print(f"\n{'='*60}")
        print("FAILURE DETAILS")
        print("=" * 60)
        
        for f in failures:
            print(f"\nQuestion: {f['question']}")
            print(f"Expected: {f['keywords']}")
            print(f"Found: {f['found']}")
            print(f"Missing: {f['missing']}")
            print("Retrieved:")
            for i, r in enumerate(f['results'], 1):
                print(f"  [{i}] {r['source']} (sim={r['similarity']:.4f})")
                print(f"      Q: {r['question'][:60]}...")
                print(f"      A: {r['answer'][:80]}...")
    
    # Verdict
    print("\n" + "=" * 60)
    if passed >= 8:
        print("RETRIEVAL QUALITY: GOOD")
        print("Ready to test full pipeline")
    elif passed >= 5:
        print("RETRIEVAL QUALITY: ACCEPTABLE")
        print("May need to review failed cases")
    else:
        print("RETRIEVAL QUALITY: POOR")
        print("Check embedding process and data")
    print("=" * 60)


if __name__ == "__main__":
    main()
