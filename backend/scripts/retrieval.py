"""
Retrieval module for Quantum Computing RAG.
Semantic search using Voyage AI embeddings and Neon pgvector.

Usage:
    from retrieval import Retriever
    
    retriever = Retriever()
    results = retriever.search("What is a qubit?", top_k=5)
"""

import os
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv
import voyageai
import psycopg2

# Load environment variables
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Settings
EMBEDDING_MODEL = "voyage-3.5-lite"
EMBEDDING_DIM = 1024


class Retriever:
    """Semantic retrieval using Voyage AI and Neon pgvector."""
    
    def __init__(self):
        """Initialize retriever with API clients."""
        self.api_key = os.getenv("VOYAGE_API_KEY")
        self.db_url = os.getenv("DATABASE_URL")
        
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY not found in environment")
        if not self.db_url:
            raise ValueError("DATABASE_URL not found in environment")
        
        self.voyage = voyageai.Client(api_key=self.api_key)
    
    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.db_url)
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            query: Search query string
        
        Returns:
            Embedding vector (1024 dimensions)
        """
        result = self.voyage.embed(
            texts=[query],
            model=EMBEDDING_MODEL,
            input_type="query"  # Critical: marks as query for retrieval
        )
        return result.embeddings[0]
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar Q&A pairs.
        
        Args:
            query: Search query string
            top_k: Number of results to return
        
        Returns:
            List of dicts with keys: question, answer, source, similarity
        """
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Search in Neon
        conn = self._get_connection()
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
                "source": row[0],
                "question": row[1],
                "answer": row[2],
                "similarity": float(row[3])
            })
        
        cur.close()
        conn.close()
        
        return results
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        conn = self._get_connection()
        cur = conn.cursor()
        
        # Total count
        cur.execute("SELECT COUNT(*) FROM chunks")
        total = cur.fetchone()[0]
        
        # Count by source
        cur.execute("""
            SELECT source, COUNT(*) 
            FROM chunks 
            GROUP BY source
        """)
        by_source = {row[0]: row[1] for row in cur.fetchall()}
        
        cur.close()
        conn.close()
        
        return {
            "total": total,
            "by_source": by_source
        }


def test_retrieval():
    """Test retrieval with sample queries."""
    print("=" * 60)
    print("RETRIEVAL TEST")
    print("=" * 60)
    
    retriever = Retriever()
    
    # Show stats
    stats = retriever.get_stats()
    print(f"\nDatabase stats:")
    print(f"  Total chunks: {stats['total']:,}")
    for source, count in stats['by_source'].items():
        print(f"    {source}: {count:,}")
    
    # Test queries
    test_queries = [
        "What is a qubit?",
        "How does quantum entanglement work?",
        "What is superposition?",
        "What is Shor's algorithm?",
        "What is quantum error correction?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)
        
        results = retriever.search(query, top_k=3)
        
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] {r['source']} (sim={r['similarity']:.4f})")
            print(f"    Q: {r['question'][:80]}...")
            print(f"    A: {r['answer'][:100]}...")


if __name__ == "__main__":
    test_retrieval()
