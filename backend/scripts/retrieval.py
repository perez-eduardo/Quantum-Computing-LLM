"""
Retrieval module for Quantum Computing RAG.
Semantic search using Voyage AI embeddings and Neon pgvector.
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
import voyageai
import psycopg2

load_dotenv()

EMBEDDING_MODEL = "voyage-3.5-lite"


class Retriever:
    def __init__(self):
        self.api_key = os.getenv("VOYAGE_API_KEY")
        self.db_url = os.getenv("DATABASE_URL")
        
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY not found")
        if not self.db_url:
            raise ValueError("DATABASE_URL not found")
        
        self.voyage = voyageai.Client(api_key=self.api_key)
    
    def _get_connection(self):
        return psycopg2.connect(self.db_url)
    
    def embed_query(self, query: str) -> List[float]:
        result = self.voyage.embed(
            texts=[query],
            model=EMBEDDING_MODEL,
            input_type="query"
        )
        return result.embeddings[0]
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.embed_query(query)
        
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
