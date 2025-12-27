# Quantum Computing Assistant - Implementation Plan

**Last Updated:** December 27, 2025

**Related Documents:**
- Design Document: `quantum-computing-assistant-design.md`
- Infrastructure Planning: `initial-exploratory-brainstorming.md`
- Model Investigation Report: `model_investigation_report.md`
- Groq Integration Plan: `groq-integration-plan.md`

---

## Current Status

**Phase 1:** Training Pipeline - ✅ COMPLETE (v5 trained, 100% pass rate)
**Phase 2:** RAG System - ✅ COMPLETE (100% retrieval accuracy)
**Phase 3:** Backend - ✅ COMPLETE (FastAPI with lazy loading)
**Phase 4:** Frontend - ✅ COMPLETE (Flask + Jinja)
**Phase 5:** Deployment - ✅ COMPLETE (Railway, live)
**Phase 6:** Groq Integration - ✅ COMPLETE (tested locally)

**Live URLs:**
- Frontend: https://quantum-computing-llm.up.railway.app
- Backend: https://quantum-computing-llm-backend.up.railway.app
- API Docs: https://quantum-computing-llm-backend.up.railway.app/docs

---

## Project Structure

```
Quantum-Computing-LLM/
├── Docs/
│   ├── implementation-plan.md
│   ├── initial-exploratory-brainstorming.md
│   ├── model_investigation_report.md
│   ├── quantum-computing-assistant-design.md
│   └── groq-integration-plan.md
│
├── training/
│   ├── model/
│   │   ├── final_model.pt              # 125.8M params (510MB, Git LFS)
│   │   └── config.json
│   ├── tokenizer/
│   │   └── tokenizer.json              # 16K vocab BPE
│   └── scripts/
│       ├── model.py                    # QuantumLLM architecture
│       ├── dataset.py                  # DataLoader classes
│       ├── train.py                    # Training script
│       └── evaluate.py                 # Evaluation script
│
├── backend/
│   ├── Dockerfile                      # CPU-only PyTorch, Git LFS pull
│   ├── Procfile                        # uvicorn startup
│   ├── requirements.txt                # tokenizers==0.22.1, groq>=0.11.0
│   ├── scripts/
│   │   ├── retrieval.py                # Retriever class (Voyage + Neon)
│   │   ├── base_inference.py           # BaseLLM abstract class
│   │   ├── inference.py                # QuantumInference(BaseLLM)
│   │   └── groq_inference.py           # GroqInference(BaseLLM)
│   └── app/
│       ├── __init__.py
│       ├── config.py                   # Environment variables + Groq config
│       └── main.py                     # Endpoints, lazy loading, LLM selection
│
├── frontend/
│   ├── Dockerfile                      # Python 3.11-slim
│   ├── Procfile                        # gunicorn --timeout 600
│   ├── app.py                          # Flask server
│   ├── requirements.txt                # flask, requests, gunicorn
│   ├── static/
│   │   ├── style.css
│   │   └── loading.gif
│   └── templates/
│       └── index.html                  # Jinja template with JS
│
├── data/
│   ├── raw/
│   └── processed/
│
├── .env                                # API keys (local only)
├── .gitattributes                      # Git LFS for *.pt files
└── requirements.txt
```

---

## Run Commands

### Local Development (Docker)

```powershell
cd E:\Personal_projects\Quantum-Computing-LLM
docker build -f backend/Dockerfile -t quantum-backend .
docker run -p 8000:8000 --env-file .env -e PORT=8000 quantum-backend
```

### Test Commands (PowerShell)

```powershell
# Health check
Invoke-WebRequest -Uri http://localhost:8000/health

# Groq mode (fast, ~725ms)
Invoke-WebRequest -Uri http://localhost:8000/query -Method POST -ContentType "application/json" -Body '{"question": "What is a qubit?", "use_groq": true}'

# Custom model (slow, ~50-80s)
Invoke-WebRequest -Uri http://localhost:8000/query -Method POST -ContentType "application/json" -Body '{"question": "What is a qubit?", "use_groq": false}'
```

### Production (Railway)

Both services deploy automatically on git push.

---

## Existing Backend Code

### Retriever (`backend/scripts/retrieval.py`)

```python
class Retriever:
    def embed_query(query: str) -> List[float]    # Voyage AI, input_type="query"
    def search(query: str, top_k: int) -> List[Dict]  # Returns question, answer, source, similarity
    def get_stats() -> Dict                        # Database statistics
```

### BaseLLM (`backend/scripts/base_inference.py`)

```python
class BaseLLM(ABC):
    @property
    def name(self) -> str                          # "groq" or "custom"
    def generate(context: str, question: str) -> str
    def extract_answer(generated_text: str) -> str
```

### QuantumInference (`backend/scripts/inference.py`)

```python
class QuantumInference(BaseLLM):
    def __init__(model_path, tokenizer_path, device)  # Loads model + tokenizer
    def generate(context, question) -> str         # Builds flat prompt internally
    def extract_answer(generated_text) -> str      # Gets first answer after "Answer:"
    @property
    def name(self) -> str                          # Returns "custom"
```

### GroqInference (`backend/scripts/groq_inference.py`)

```python
class GroqInference(BaseLLM):
    def __init__()                                 # Initializes Groq client
    def generate(context, question) -> str         # Uses chat completion API
    def extract_answer(generated_text) -> str      # Returns text as-is (already clean)
    @property
    def name(self) -> str                          # Returns "groq"
```

---

## Architecture

### Two LLM Modes

| Mode | LLM | Speed | Use Case |
|------|-----|-------|----------|
| Production | Groq API (Llama 3.3 70B) | ~725ms | Fast UX for users |
| Demo | Custom 125.8M | ~50-80s | Prove ML skills to recruiters |

### API Request/Response

**Request:**
```json
{
  "question": "What is a qubit?",
  "use_groq": true
}
```

**Response:**
```json
{
  "answer": "A qubit, short for quantum bit...",
  "sources": [...],
  "response_time_ms": 725,
  "model_loaded_fresh": false,
  "suggested_question": "can you explain qubits in simple terms?",
  "llm_used": "groq"
}
```

### Pipeline

```
User Question → Voyage AI embed → Neon vector search → Build prompt → LLM generates answer
                                                                       ↓
                                                            use_groq=true  → Groq API (~725ms)
                                                            use_groq=false → Custom model (~50-80s)
```

### Custom Model Config

| Parameter | Value |
|-----------|-------|
| Temperature | 0.2 |
| Top-k | 30 |
| Loading | Lazy (load on first request) |
| Timeout | Unload after 5 min idle |

### Groq Config

| Parameter | Value |
|-----------|-------|
| Model | llama-3.3-70b-versatile |
| Temperature | 0.2 |
| Max tokens | 300 |
| Loading | Always loaded (lightweight client) |

---

## Parameter Tuning (December 26, 2025)

### Battery Test on HPC

Tested 24 parameter combinations (4 temps × 6 top_k) across 20 questions = 480 tests.

**Top Results:**

| Parameters | Pass Rate | Keyword Score |
|------------|-----------|---------------|
| **temp=0.2, top_k=30** | **100%** | **80.5%** |
| temp=0.4, top_k=20 | 100% | 78.8% |
| temp=0.1, top_k=40 | 100% | 78.5% |
| temp=0.3, top_k=50 (old baseline) | 100% | 74.2% |

### Live Deployment Verification

Full RAG pipeline test (Voyage API + Neon DB + Custom Model):

| Parameters | Pass Rate | Keyword Score | Avg Time |
|------------|-----------|---------------|----------|
| **temp=0.2, top_k=30** | **100%** | 76.2% | 36.6s |
| temp=0.4, top_k=20 | 95% | 76.5% | 39.0s |

**Winner:** temp=0.2, top_k=30

---

## RAG System Fixed (December 25, 2025)

### Index Issue
IVFFlat approximate index was missing exact matches.

### Fix
Removed IVFFlat index, using exact search. 28K rows searches in ~300ms.

### Results

| Metric | Before | After |
|--------|--------|-------|
| Retrieval accuracy | 90% | **100%** |
| Search time | ~50ms | ~300ms |

---

## What Has Been Done

### Data Acquisition

| Task | Status | Notes |
|------|--------|-------|
| Download Stack Exchange QC dump | ✅ Done | 28K total posts |
| Obtain book PDFs | ✅ Done | 5 books |
| Generate Claude Q&A | ✅ Done | 15,000 pairs across 38 batches |
| Obtain CoT Reasoning Dataset | ✅ Done | 2,998 Q&A pairs |
| Clean books | ✅ Done | 620K words |

### Data Processing

| Task | Status | Output |
|------|--------|--------|
| Process Stack Exchange XML | ✅ Done | 10,673 pairs |
| Generate Claude Q&A | ✅ Done | 15,000 pairs |
| Process CoT dataset | ✅ Done | 2,998 pairs |
| Add context to all datasets | ✅ Done | Topic-matched Q&A pairs |

### RAG System

| Task | Status | Notes |
|------|--------|-------|
| Set up Neon database with pgvector | ✅ Done | Free tier |
| Embed Q&A pairs | ✅ Done | 28,071 pairs |
| Fix index issue | ✅ Done | Exact search |
| Test retrieval | ✅ Done | **100% pass rate** |
| Create Retriever class | ✅ Done | `backend/scripts/retrieval.py` |
| Create QuantumInference class | ✅ Done | `backend/scripts/inference.py` |

### Custom Model Training

| Task | Status | Notes |
|------|--------|-------|
| Train 125.8M transformer | ✅ Done | Two-phase training |
| Tune generation params | ✅ Done | temp=0.2, top_k=30 |
| Achieve 100% pass rate | ✅ Done | Best config |
| Fix extraction function | ✅ Done | First answer, not last |

### Phase 3: Backend (FastAPI)

| Task | Status | Notes |
|------|--------|-------|
| Create `backend/app/config.py` | ✅ Done | Env vars, model paths |
| Create `backend/app/main.py` | ✅ Done | Endpoints, lazy loading |
| Implement lazy model loading | ✅ Done | 5 min idle timeout |
| Test `/query` endpoint | ✅ Done | 50-80s response time |
| Suggested question feature | ✅ Done | From retrieved results |

### Phase 4: Frontend (Flask + Jinja)

| Task | Status | Notes |
|------|--------|-------|
| Create Flask app | ✅ Done | Proxies to backend API |
| Welcome screen | ✅ Done | Atom animation, 5 starter questions |
| Chat interface | ✅ Done | User/AI message bubbles |
| Loading indicator | ✅ Done | Rotating messages + GIF |
| Free-tier disclaimer | ✅ Done | 40-90s warning |
| Suggested follow-up button | ✅ Done | Clickable next question |

### Phase 5: Deployment (Railway)

| Task | Status | Notes |
|------|--------|-------|
| Create Railway project | ✅ Done | Hobby plan ($5/month) |
| Configure backend service | ✅ Done | Dockerfile, env vars |
| Configure frontend service | ✅ Done | Dockerfile, BACKEND_URL |
| Set up Git LFS | ✅ Done | Model pulled during build |
| Fix tokenizers version | ✅ Done | 0.15.0 → 0.22.1 |
| Fix gunicorn timeout | ✅ Done | 30s → 600s |
| Test production endpoints | ✅ Done | Working end-to-end |

### Phase 6: Groq Integration

| Task | Status | Notes |
|------|--------|-------|
| Create BaseLLM abstract class | ✅ Done | `backend/scripts/base_inference.py` |
| Update QuantumInference | ✅ Done | Inherits BaseLLM |
| Create GroqInference class | ✅ Done | `backend/scripts/groq_inference.py` |
| Add Groq config | ✅ Done | `backend/app/config.py` |
| Update API models | ✅ Done | `use_groq`, `llm_used` fields |
| Add groq dependency | ✅ Done | groq>=0.11.0 |
| Delete unused pipeline.py | ✅ Done | Was not being used |
| Test locally with Docker | ✅ Done | 20 queries passed |
| Add GROQ_API_KEY to Railway | ⬜ Pending | |
| Deploy to Railway | ⬜ Pending | |

---

## What Is Next

### Immediate

| Task | Priority | Status |
|------|----------|--------|
| Add GROQ_API_KEY to Railway | High | ⬜ Pending |
| Deploy Groq integration | High | ⬜ Pending |
| Add frontend toggle for LLM mode | Medium | ⬜ Pending |

### Future Improvements

| Task | Priority | Status |
|------|----------|--------|
| Implement response caching | Low | ⬜ Pending |
| Add request queuing | Low | ⬜ Pending |
| Set up monitoring/alerting | Low | ⬜ Pending |

---

## Key Findings

1. **ChatGPT synthetic data was 94% garbage.** Abandoned entirely.

2. **Claude Q&A generation works.** 15,000 pairs verified.

3. **IVFFlat index caused retrieval failures.** Exact search fixed it.

4. **Custom model best params:** temp=0.2, top_k=30 (100% pass rate, 76-80% keyword score).

5. **Extraction function bug:** Was using rfind() (last answer), fixed to find() (first answer).

6. **Lazy loading saves cost.** ~$2-3/month vs $6-8/month always loaded.

7. **Custom model inference:** ~50-80s per question on Railway CPU.

8. **HPC battery test:** 480 tests in 5.8 minutes on H100 (vs ~280 min on CPU).

9. **RAG pipeline classes exist.** Retriever, QuantumInference, GroqInference ready to use.

10. **Flask + Jinja frontend.** Simpler than React, single Python file with Jinja templates.

11. **Git LFS requires explicit pull in Docker.** Railway doesn't auto-pull LFS files.

12. **Tokenizers version must match.** Training used 0.22.1, deployment initially had 0.15.0.

13. **Gunicorn default timeout (30s) too short.** Set to 600s for model loading + inference.

14. **Groq integration works.** ~725ms response time, 20/20 test queries passed.

15. **groq package version matters.** 0.4.2 had httpx proxy issues, 0.11.0+ works.

---

*Document version: 22.0*
