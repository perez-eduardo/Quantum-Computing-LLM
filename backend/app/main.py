"""
FastAPI application for Quantum Computing LLM.
Supports both custom model (lazy loaded) and Groq API.

Run:
    cd backend
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import sys
import time
import asyncio
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
from inference import QuantumInference
from groq_inference import GroqInference

from app.config import (
    MODEL_PATH,
    TOKENIZER_PATH,
    MODEL_TEMPERATURE,
    MODEL_TOP_K,
    MODEL_MAX_NEW_TOKENS,
    IDLE_TIMEOUT_SECONDS,
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    GROQ_TEMPERATURE,
    GROQ_MAX_TOKENS,
    validate_config,
    validate_groq_config,
)


# Global state for lazy loading (custom model only)
class ModelState:
    def __init__(self):
        self.inference: Optional[QuantumInference] = None
        self.last_access: Optional[float] = None
        self.loading: bool = False


model_state = ModelState()
retriever: Optional[Retriever] = None
groq_inference: Optional[GroqInference] = None


def get_custom_model() -> QuantumInference:
    """Get custom model instance, loading if necessary."""
    if model_state.inference is None:
        if model_state.loading:
            raise HTTPException(status_code=503, detail="Model is loading, please wait")
        
        model_state.loading = True
        try:
            print(f"Loading custom model from {MODEL_PATH}...")
            start = time.time()
            model_state.inference = QuantumInference(
                model_path=str(MODEL_PATH),
                tokenizer_path=str(TOKENIZER_PATH),
                device="cpu",
                temperature=MODEL_TEMPERATURE,
                top_k=MODEL_TOP_K,
                max_new_tokens=MODEL_MAX_NEW_TOKENS
            )
            elapsed = time.time() - start
            print(f"Custom model loaded in {elapsed:.1f}s")
        except Exception as e:
            print(f"Custom model load failed: {type(e).__name__}: {e}")
            raise
        finally:
            model_state.loading = False
    
    model_state.last_access = time.time()
    return model_state.inference


def get_groq_model() -> GroqInference:
    """Get Groq instance (singleton, always available)."""
    global groq_inference
    
    if groq_inference is None:
        validate_groq_config()
        groq_inference = GroqInference(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL_NAME,
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS
        )
    
    return groq_inference


def unload_custom_model():
    """Unload custom model to free memory."""
    if model_state.inference is not None:
        print("Unloading custom model due to idle timeout...")
        model_state.inference = None
        model_state.last_access = None


async def idle_checker():
    """Background task to unload custom model after idle timeout."""
    while True:
        await asyncio.sleep(60)  # Check every minute
        
        if model_state.inference is not None and model_state.last_access is not None:
            idle_time = time.time() - model_state.last_access
            if idle_time > IDLE_TIMEOUT_SECONDS:
                unload_custom_model()


def text_similarity(a: str, b: str) -> float:
    """Calculate text similarity ratio (0-1)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def get_suggested_question(
    original_question: str,
    answer: str,
    retrieved_results: List[dict]
) -> Optional[str]:
    """
    Pick a suggested follow-up question from retrieved results.
    
    Logic:
    1. Skip questions too similar to original (>60% similarity)
    2. Prefer questions that mention terms from the answer
    3. Return the best candidate
    """
    if not retrieved_results:
        return None
    
    # Extract key terms from answer (words > 5 chars, likely meaningful)
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
        
        # Skip if too similar to original question
        similarity = text_similarity(original_question, question)
        if similarity > 0.6:
            continue
        
        # Score by how many answer terms appear in this question
        question_lower = question.lower()
        term_matches = sum(1 for word in answer_words if word in question_lower)
        
        candidates.append({
            "question": question,
            "similarity": similarity,
            "term_matches": term_matches,
        })
    
    if not candidates:
        return None
    
    # Sort by term_matches (desc), then by similarity (asc, prefer less similar)
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
    
    # Startup
    print("Starting Quantum Computing LLM API...")
    validate_config()
    
    # Initialize retriever (lightweight, always loaded)
    retriever = Retriever()
    print("Retriever initialized")
    
    # Check Groq availability
    if GROQ_API_KEY:
        print("Groq API key found, Groq mode available")
    else:
        print("Groq API key not found, Groq mode disabled")
    
    # Start idle checker background task
    task = asyncio.create_task(idle_checker())
    
    yield
    
    # Shutdown
    task.cancel()
    unload_custom_model()
    print("API shutdown complete")


app = FastAPI(
    title="Quantum Computing LLM API",
    description="RAG-powered quantum computing Q&A with custom model and Groq API",
    version="2.0.0",
    lifespan=lifespan,
)


# Request/Response models
class QueryRequest(BaseModel):
    question: str
    use_groq: bool = False


class Source(BaseModel):
    question: str
    source: str
    similarity: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    response_time_ms: int
    model_loaded_fresh: bool
    suggested_question: Optional[str]
    llm_used: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    idle_seconds: Optional[int]
    groq_available: bool


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    idle_seconds = None
    if model_state.last_access is not None:
        idle_seconds = int(time.time() - model_state.last_access)
    
    return HealthResponse(
        status="ok",
        model_loaded=model_state.inference is not None,
        idle_seconds=idle_seconds,
        groq_available=bool(GROQ_API_KEY),
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Answer a quantum computing question."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    start_time = time.time()
    was_loaded = model_state.inference is not None
    
    # Step 1: Retrieve context
    results = retriever.search(request.question, top_k=5)
    
    if not results:
        raise HTTPException(status_code=404, detail="No relevant context found")
    
    # Step 2: Build context from top 3 results
    context = build_context(results, top_k=3)
    
    # Step 3: Select LLM and generate
    if request.use_groq:
        try:
            llm = get_groq_model()
        except ValueError as e:
            raise HTTPException(status_code=503, detail=str(e))
        model_loaded_fresh = False  # Groq is always ready
    else:
        llm = get_custom_model()
        model_loaded_fresh = not was_loaded
    
    # Step 4: Generate answer
    generated = llm.generate(context, request.question)
    answer = llm.extract_answer(generated)
    
    # Step 5: Get suggested follow-up question
    suggested = get_suggested_question(request.question, answer, results)
    
    # Step 6: Format response
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
        model_loaded_fresh=model_loaded_fresh,
        suggested_question=suggested,
        llm_used=llm.name,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
