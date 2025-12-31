"""FastAPI application for Quantum Computing LLM (Dual-mode: Groq + Custom)."""

import sys
import time
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager
from difflib import SequenceMatcher

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

SCRIPTS_PATH = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_PATH))

from retrieval import Retriever
from groq_inference import GroqInference
from modal_inference import ModalInference
from app.config import (
    GROQ_API_KEY, GROQ_MODEL_NAME, GROQ_TEMPERATURE, GROQ_MAX_TOKENS,
    MODAL_URL, validate_config
)

retriever: Optional[Retriever] = None
groq_inference: Optional[GroqInference] = None
modal_inference: Optional[ModalInference] = None


def get_groq() -> GroqInference:
    global groq_inference
    if groq_inference is None:
        groq_inference = GroqInference(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL_NAME,
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS
        )
    return groq_inference


def get_modal() -> ModalInference:
    global modal_inference
    if modal_inference is None:
        modal_inference = ModalInference(url=MODAL_URL)
    return modal_inference


def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def get_suggested_question(original: str, answer: str, results: List[dict]) -> Optional[str]:
    if not results:
        return None
    
    answer_words = set(w.lower().strip(".,!?()[]{}:;\"'") for w in answer.split() if len(w) > 5)
    candidates = []
    
    for r in results:
        q = r.get("question", "").strip()
        if not q:
            continue
        sim = text_similarity(original, q)
        if sim > 0.6:
            continue
        matches = sum(1 for w in answer_words if w in q.lower())
        candidates.append({"question": q, "similarity": sim, "matches": matches})
    
    if not candidates:
        return None
    
    candidates.sort(key=lambda x: (-x["matches"], x["similarity"]))
    return candidates[0]["question"]


def build_context(results: List[dict], top_k: int = 3) -> str:
    parts = [f"Q: {r['question']} A: {r['answer'][:300]}" for r in results[:top_k]]
    return " ".join(parts)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    print("Starting Quantum Computing LLM API...")
    validate_config()
    retriever = Retriever()
    print(f"Modal URL: {MODAL_URL}")
    print("Ready")
    yield
    print("Shutdown")


app = FastAPI(title="Quantum Computing LLM API", version="4.0.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    model: str = "groq"  # "groq" or "custom"


class Source(BaseModel):
    question: str
    source: str
    similarity: float


class QueryResponse(BaseModel):
    model_config = {'protected_namespaces': ()}
    answer: str
    sources: List[Source]
    response_time_ms: int
    suggested_question: Optional[str]
    model_used: str


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok")


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    start = time.time()
    results = retriever.search(request.question, top_k=5)
    
    if not results:
        raise HTTPException(status_code=404, detail="No relevant context found")
    
    context = build_context(results, top_k=3)
    
    # Route to appropriate model
    if request.model == "custom":
        print(f"Using Custom Model (Modal)")
        llm = get_modal()
        model_used = "custom"
    else:
        print(f"Using Groq")
        llm = get_groq()
        model_used = "groq"
    
    answer = llm.generate(context, request.question)
    suggested = get_suggested_question(request.question, answer, results)
    elapsed_ms = int((time.time() - start) * 1000)
    
    sources = [
        Source(question=r["question"][:100], source=r["source"], similarity=round(r["similarity"], 4))
        for r in results[:3]
    ]
    
    return QueryResponse(
        answer=answer,
        sources=sources,
        response_time_ms=elapsed_ms,
        suggested_question=suggested,
        model_used=model_used
    )

