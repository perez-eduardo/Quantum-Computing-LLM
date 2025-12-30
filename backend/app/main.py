"""
FastAPI application for Quantum Computing LLM.
Groq-only version.
"""

import sys
import time
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager
from difflib import SequenceMatcher

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add scripts folder to path for imports
SCRIPTS_PATH = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_PATH))

from retrieval import Retriever
from groq_inference import GroqInference

from app.config import (
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    GROQ_TEMPERATURE,
    GROQ_MAX_TOKENS,
    validate_config,
)

# Global state
retriever: Optional[Retriever] = None
groq_inference: Optional[GroqInference] = None


def get_groq() -> GroqInference:
    """Get Groq instance (singleton)."""
    global groq_inference
    
    if groq_inference is None:
        groq_inference = GroqInference(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL_NAME,
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS
        )
    
    return groq_inference


def text_similarity(a: str, b: str) -> float:
    """Calculate text similarity ratio (0-1)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def get_suggested_question(
    original_question: str,
    answer: str,
    retrieved_results: List[dict]
) -> Optional[str]:
    """Pick a suggested follow-up question from retrieved results."""
    if not retrieved_results:
        return None
    
    answer_words = set(
        word.lower().strip(".,!?()[]{}:;\"'")
        for word in answer.split()
        if len(word) > 5
    )
    
    candidates = []
    
    for result in retrieved_results:
        question = result.get("question", "").strip()
        if not question:
            continue
        
        similarity = text_similarity(original_question, question)
        if similarity > 0.6:
            continue
        
        question_lower = question.lower()
        term_matches = sum(1 for word in answer_words if word in question_lower)
        
        candidates.append({
            "question": question,
            "similarity": similarity,
            "term_matches": term_matches,
        })
    
    if not candidates:
        return None
    
    candidates.sort(key=lambda x: (-x["term_matches"], x["similarity"]))
    
    return candidates[0]["question"]


def build_context(results: List[dict], top_k: int = 3) -> str:
    """Build context string from retrieved Q&A pairs."""
    context_parts = []
    for r in results[:top_k]:
        q = r["question"]
        a = r["answer"][:300]
        context_parts.append(f"Q: {q} A: {a}")
    return " ".join(context_parts)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global retriever
    
    print("Starting Quantum Computing LLM API...")
    validate_config()
    
    retriever = Retriever()
    print("Retriever initialized")
    print("Groq mode ready")
    
    yield
    
    print("API shutdown complete")


app = FastAPI(
    title="Quantum Computing LLM API",
    description="RAG-powered quantum computing Q&A with Groq",
    version="3.0.0",
    lifespan=lifespan,
)


# Request/Response models
class QueryRequest(BaseModel):
    question: str


class Source(BaseModel):
    question: str
    source: str
    similarity: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    response_time_ms: int
    suggested_question: Optional[str]


class HealthResponse(BaseModel):
    status: str


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Answer a quantum computing question."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    start_time = time.time()
    
    # Step 1: Retrieve context
    results = retriever.search(request.question, top_k=5)
    
    if not results:
        raise HTTPException(status_code=404, detail="No relevant context found")
    
    # Step 2: Build context from top 3 results
    context = build_context(results, top_k=3)
    
    # Step 3: Generate with Groq
    llm = get_groq()
    generated = llm.generate(context, request.question)
    answer = llm.extract_answer(generated)
    
    # Step 4: Get suggested follow-up question
    suggested = get_suggested_question(request.question, answer, results)
    
    # Step 5: Format response
    elapsed_ms = int((time.time() - start_time) * 1000)
    
    sources = [
        Source(
            question=r["question"][:100],
            source=r["source"],
            similarity=round(r["similarity"], 4),
        )
        for r in results[:3]
    ]
    
    return QueryResponse(
        answer=answer,
        sources=sources,
        response_time_ms=elapsed_ms,
        suggested_question=suggested,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
