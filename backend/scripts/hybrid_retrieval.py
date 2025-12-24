"""
Hybrid Retrieval Service: Semantic + Full-Text Search with RRF fusion.

Combines Voyage AI embeddings with PostgreSQL full-text search
for improved retrieval accuracy.
"""

import os
from typing import Optional
import voyageai
import psycopg2
from psycopg2.extras import RealDictCursor

EMBEDDING_MODEL = "voyage-3.5-lite"
RRF_K = 60  # Standard RRF constant
SEMANTIC_WEIGHT = 0.5  # Balance between semantic and keyword search


class HybridRetriever:
    def __init__(self):
        self.voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        self.db_url = os.getenv("DATABASE_URL")

    def _get_connection(self):
        return psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)

    def _semantic_search(self, query: str, limit: int = 20) -> list:
        """Vector similarity search using Voyage embeddings."""
        # Embed query
        result = self.voyage.embed(
            texts=[query],
            model=EMBEDDING_MODEL,
            input_type="query"
        )
        query_embedding = result.embeddings[0]

        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT id, book_name, chunk_index, content,
                   1 - (embedding <=> %s::vector) as similarity
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, limit))

        results = cur.fetchall()
        cur.close()
        conn.close()

        return [dict(r) for r in results]

    def _keyword_search(self, query: str, limit: int = 20) -> list:
        """Full-text search using PostgreSQL tsvector."""
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT id, book_name, chunk_index, content,
                   ts_rank(content_tsv, plainto_tsquery('english', %s)) as rank
            FROM chunks
            WHERE content_tsv @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
        """, (query, query, limit))

        results = cur.fetchall()
        cur.close()
        conn.close()

        return [dict(r) for r in results]

    def _rrf_fusion(self, semantic_results: list, keyword_results: list, 
                    top_k: int, alpha: float = SEMANTIC_WEIGHT) -> list:
        """
        Reciprocal Rank Fusion to combine semantic and keyword results.
        
        score = α * (1 / (k + semantic_rank)) + (1-α) * (1 / (k + keyword_rank))
        """
        scores = {}
        content_map = {}

        # Score semantic results
        for rank, result in enumerate(semantic_results):
            chunk_id = result['id']
            scores[chunk_id] = alpha * (1 / (RRF_K + rank))
            content_map[chunk_id] = result

        # Add keyword results
        for rank, result in enumerate(keyword_results):
            chunk_id = result['id']
            keyword_score = (1 - alpha) * (1 / (RRF_K + rank))
            
            if chunk_id in scores:
                scores[chunk_id] += keyword_score
            else:
                scores[chunk_id] = keyword_score
                content_map[chunk_id] = result

        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Build final results
        results = []
        for chunk_id in sorted_ids[:top_k]:
            item = content_map[chunk_id]
            item['rrf_score'] = scores[chunk_id]
            results.append(item)

        return results

    def search(self, query: str, top_k: int = 3, 
               alpha: Optional[float] = None) -> list:
        """
        Hybrid search combining semantic and keyword search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for semantic vs keyword (0-1). 
                   Higher = more semantic, Lower = more keyword.
                   Default is 0.5 (equal weight).
        
        Returns:
            List of chunk dictionaries with rrf_score
        """
        if alpha is None:
            alpha = SEMANTIC_WEIGHT

        # Get results from both methods (fetch more for fusion)
        fetch_limit = top_k * 5
        
        semantic_results = self._semantic_search(query, limit=fetch_limit)
        keyword_results = self._keyword_search(query, limit=fetch_limit)

        # Fuse results
        combined = self._rrf_fusion(
            semantic_results, 
            keyword_results, 
            top_k=top_k,
            alpha=alpha
        )

        return combined

    def search_semantic_only(self, query: str, top_k: int = 3) -> list:
        """Pure semantic search (for comparison)."""
        results = self._semantic_search(query, limit=top_k)
        return results

    def search_keyword_only(self, query: str, top_k: int = 3) -> list:
        """Pure keyword search (for comparison)."""
        results = self._keyword_search(query, limit=top_k)
        return results


# Singleton instance
_retriever = None

def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Main retrieval function for RAG pipeline.
    Returns formatted context string.
    """
    retriever = get_retriever()
    results = retriever.search(query, top_k=top_k)

    if not results:
        return ""

    context_parts = []
    for i, chunk in enumerate(results, 1):
        source = chunk.get('book_name', 'unknown')
        content = chunk.get('content', '')
        context_parts.append(f"[Source {i}: {source}]\n{content}")

    return "\n\n".join(context_parts)
