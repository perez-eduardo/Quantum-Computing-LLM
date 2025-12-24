"""
RAG retrieval: Query similar chunks from Neon DB.
Run from project root: python backend/scripts/retrieval.py

Can be imported as a module or run standalone for testing.
"""

import os
from dotenv import load_dotenv
import voyageai
import psycopg2

load_dotenv()

# Settings
EMBEDDING_MODEL = "voyage-3.5-lite"
DEFAULT_TOP_K = 5


class Retriever:
    """RAG retrieval class for querying similar chunks."""
    
    def __init__(self):
        api_key = os.getenv("VOYAGE_API_KEY")
        db_url = os.getenv("DATABASE_URL")
        
        if not api_key:
            raise ValueError("VOYAGE_API_KEY not found in .env")
        if not db_url:
            raise ValueError("DATABASE_URL not found in .env")
        
        self.voyage_client = voyageai.Client(api_key=api_key)
        self.db_url = db_url
    
    def embed_query(self, query):
        """Generate embedding for a query."""
        result = self.voyage_client.embed(
            texts=[query],
            model=EMBEDDING_MODEL,
            input_type="query"  # CRITICAL: marks this as a query for retrieval
        )
        return result.embeddings[0]
    
    def search(self, query, top_k=DEFAULT_TOP_K):
        """
        Search for similar chunks.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of dicts with keys: book_name, chunk_index, content, similarity
        """
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Search in Neon
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT 
                book_name,
                chunk_index,
                content,
                1 - (embedding <=> %s::vector) as similarity
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, top_k))
        
        results = []
        for row in cur.fetchall():
            results.append({
                'book_name': row[0],
                'chunk_index': row[1],
                'content': row[2],
                'similarity': float(row[3])
            })
        
        cur.close()
        conn.close()
        
        return results


def test_retrieval():
    """Test retrieval with sample queries."""
    print("=" * 40)
    print("TEST RETRIEVAL")
    print("=" * 40)
    
    retriever = Retriever()
    
    test_queries = [
        "What is a qubit?",
        "How does quantum entanglement work?",
        "What is superposition?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*40}")
        print(f"Query: {query}")
        print("=" * 40)
        
        results = retriever.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result['book_name']} (chunk {result['chunk_index']})")
            print(f"    Similarity: {result['similarity']:.4f}")
            print(f"    Content: {result['content'][:200]}...")


if __name__ == "__main__":
    test_retrieval()
