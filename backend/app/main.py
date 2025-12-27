"""
FastAPI application for Quantum Computing LLM.
Custom model with lazy loading (loads on first request, unloads after 5 min idle).

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

print(f"DEBUG: SCRIPTS_PATH = {SCRIPTS_PATH}")

from retrieval import Retriever
from inference import QuantumInference

from app.config import (
    MODEL_PATH,
    TOKENIZER_PATH,
    MODEL_TEMPERATURE,
    MODEL_TOP_K,
    MODEL_MAX_NEW_TOKENS,
    IDLE_TIMEOUT_SECONDS,
    validate_config,
)

print(f"DEBUG: MODEL_PATH = {MODEL_PATH}")
print(f"DEBUG: TOKENIZER_PATH = {TOKENIZER_PATH}")


# Global state for lazy loading
class ModelState:
    def __init__(self):
        self.inference: Optional[QuantumInference] = None
        self.last_access: Optional[float] = None
        self.loading: bool = False


model_state = ModelState()
retriever: Optional[Retriever] = None


def get_model() -> QuantumInference:
    """Get model instance, loading if necessary."""
    print(f"DEBUG: get_model() called")
    print(f"DEBUG: inference is None: {model_state.inference is None}")
    print(f"DEBUG: loading: {model_state.loading}")
    
    if model_state.inference is None:
        if model_state.loading:
            print("DEBUG: Model already loading, returning 503")
            raise HTTPException(status_code=503, detail="Model is loading, please wait")
        
        model_state.loading = True
        try:
            print(f"DEBUG: Starting model load...")
            print(f"DEBUG: MODEL_PATH exists: {Path(MODEL_PATH).exists()}")
            print(f"DEBUG: TOKENIZER_PATH exists: {Path(TOKENIZER_PATH).exists()}")
            
            start = time.time()
            model_state.inference = QuantumInference(
                model_path=str(MODEL_PATH),
                tokenizer_path=str(TOKENIZER_PATH),
                device="cpu"
            )
            elapsed = time.time() - start
            print(f"DEBUG: Model loaded in {elapsed:.1f}s")
        except Exception as e:
            print(f"DEBUG: Model load FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            model_state.loading = False
    
    model_state.last_access = time.time()
    return model_state.inference


def unload_model():
    """Unload model to free memory."""
    if model_state.inference is not None:
        print("Unloading model due to idle timeout...")
        model_state.inference = None
        model_state.last_access = None


async def idle_checker():
    """Background task to unload model after idle timeout."""
    while True:
        await asyncio.sleep(60)  # Check every minute
        
        if model_state.inference is not None and model_state.last_access is not None:
            idle_time = time.time() - model_state.last_access
            if idle_time > IDLE_TIMEOUT_SECONDS:
                unload_model()


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global retriever
    
    # Startup
    print("Starting Quantum Computing LLM API...")
    print(f"DEBUG: Validating config...")
    validate_config()
    print(f"DEBUG: Config validated")
    
    # Initialize retriever (lightweight, always loaded)
    print(f"DEBUG: Initializing retriever...")
    retriever = Retriever()
    print("Retriever initialized")
    
    # Start idle checker background task
    task = asyncio.create_task(idle_checker())
    print(f"DEBUG: Idle checker started")
    
    yield
    
    # Shutdown
    task.cancel()
    unload_model()
    print("API shutdown complete")


app = FastAPI(
    title="Quantum Computing LLM API",
    description="RAG-powered quantum computing Q&A with custom 125.8M parameter model",
    version="1.0.0",
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
    model_loaded_fresh: bool
    suggested_question: Optional[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    idle_seconds: Optional[int]


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
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Answer a quantum computing question."""
    print(f"DEBUG: /query endpoint hit")
    print(f"DEBUG: Question: {request.question[:100]}")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    start_time = time.time()
    was_loaded = model_state.inference is not None
    print(f"DEBUG: was_loaded = {was_loaded}")
    
    # Step 1: Retrieve context
    print("DEBUG: Step 1 - Retrieving context...")
    try:
        results = retriever.search(request.question, top_k=5)
        print(f"DEBUG: Retrieved {len(results) if results else 0} results")
    except Exception as e:
        print(f"DEBUG: Retrieval FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    if not results:
        raise HTTPException(status_code=404, detail="No relevant context found")
    
    # Step 2: Build context from top 3 results
    print("DEBUG: Step 2 - Building context...")
    context_parts = []
    for r in results[:3]:
        q = r["question"]
        a = r["answer"][:300]
        context_parts.append(f"Q: {q} A: {a}")
    context = " ".join(context_parts)
    print(f"DEBUG: Context length: {len(context)} chars")
    
    # Step 3: Build prompt
    print("DEBUG: Step 3 - Building prompt...")
    prompt = f"Context: {context} Question: {request.question} Answer:"
    print(f"DEBUG: Prompt length: {len(prompt)} chars")
    
    # Step 4: Generate answer (loads model if needed)
    print("DEBUG: Step 4 - Getting model...")
    try:
        inference = get_model()
        print("DEBUG: Got model, generating...")
        generated = inference.generate(
            prompt,
            max_new_tokens=MODEL_MAX_NEW_TOKENS,
            temperature=MODEL_TEMPERATURE,
            top_k=MODEL_TOP_K,
        )
        print(f"DEBUG: Generated {len(generated)} chars")
        answer = inference.extract_answer(generated)
        print(f"DEBUG: Extracted answer: {len(answer)} chars")
    except Exception as e:
        print(f"DEBUG: Generation FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Step 5: Get suggested follow-up question
    print("DEBUG: Step 5 - Getting suggested question...")
    suggested = get_suggested_question(request.question, answer, results)
    print(f"DEBUG: Suggested: {suggested[:50] if suggested else None}")
    
    # Step 6: Format response
    print("DEBUG: Step 6 - Formatting response...")
    elapsed_ms = int((time.time() - start_time) * 1000)
    
    sources = [
        Source(
            question=r["question"][:100],
            source=r["source"],
            similarity=round(r["similarity"], 4),
        )
        for r in results[:3]
    ]
    
    print(f"DEBUG: Done! Response time: {elapsed_ms}ms")
    
    return QueryResponse(
        answer=answer,
        sources=sources,
        response_time_ms=elapsed_ms,
        model_loaded_fresh=not was_loaded,
        suggested_question=suggested,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    