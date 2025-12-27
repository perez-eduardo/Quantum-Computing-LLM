# Quantum Computing Assistant - Design Document

## Overview

Design specification for a web-based application that answers questions about foundational quantum computing concepts. Powered by custom-trained 125.8M parameter transformer (demo mode) and Groq API with Llama 3.3 70B (production mode).

**Related Documents:**
- Infrastructure Planning: `initial-exploratory-brainstorming.md`
- Implementation Plan: `implementation-plan.md`
- Model Investigation Report: `model_investigation_report.md`
- Groq Integration Plan: `groq-integration-plan.md`

**Disclaimer:** Model weights are private and not distributed. Training data used for educational/personal purposes only.

---

## Project Goals

**Primary Goal:** Portfolio demonstration for recruiters

This project proves ability to:
- Design transformer architecture from scratch
- Build complete two-phase training pipeline
- Train on real HPC hardware (H100 GPUs)
- Deploy end-to-end ML system with RAG
- Integrate third-party APIs (Groq, Voyage AI)

**Target Audience:** Recruiters evaluating ML skills, students curious about quantum computing

---

## Current Status

| Component | Status |
|-----------|--------|
| Custom Model (v5) | ✅ COMPLETE (125.8M params, 100% pass rate) |
| RAG System | ✅ COMPLETE (100% retrieval) |
| Parameter Tuning | ✅ COMPLETE (temp=0.2, top_k=30) |
| Backend Classes | ✅ COMPLETE (Retriever, BaseLLM, QuantumInference, GroqInference) |
| FastAPI App | ✅ COMPLETE (lazy loading, suggested questions, LLM selection) |
| Frontend | ✅ COMPLETE (Flask + Jinja) |
| Deployment | ✅ COMPLETE (Railway, live) |
| Groq Integration | ✅ COMPLETE (tested locally, 20/20 queries passed) |

---

## Live URLs

| Service | URL |
|---------|-----|
| **Frontend** | https://quantum-computing-llm.up.railway.app |
| **Backend** | https://quantum-computing-llm-backend.up.railway.app |
| **API Docs** | https://quantum-computing-llm-backend.up.railway.app/docs |
| **Health Check** | https://quantum-computing-llm-backend.up.railway.app/health |

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
│   │   ├── final_model.pt              # 125.8M params (Git LFS)
│   │   └── config.json
│   ├── tokenizer/
│   │   └── tokenizer.json              # 16K vocab BPE
│   └── scripts/
│       └── model.py                    # QuantumLLM architecture
│
├── backend/
│   ├── Dockerfile                      # CPU PyTorch, Git LFS pull
│   ├── Procfile
│   ├── requirements.txt                # tokenizers==0.22.1, groq>=0.11.0
│   ├── scripts/
│   │   ├── retrieval.py                # Retriever class
│   │   ├── base_inference.py           # BaseLLM abstract class
│   │   ├── inference.py                # QuantumInference(BaseLLM)
│   │   └── groq_inference.py           # GroqInference(BaseLLM)
│   └── app/
│       ├── __init__.py
│       ├── config.py                   # Environment variables + Groq config
│       └── main.py                     # Endpoints, lazy loading, LLM selection
│
├── frontend/
│   ├── Dockerfile
│   ├── Procfile                        # gunicorn --timeout 600
│   ├── app.py                          # Flask server
│   ├── requirements.txt
│   ├── static/
│   │   ├── style.css
│   │   └── loading.gif
│   └── templates/
│       └── index.html                  # Jinja template with JS
│
└── .env                                # API keys (local only)
```

---

## Run Commands

**Docker (Recommended):**
```powershell
cd E:\Personal_projects\Quantum-Computing-LLM
docker build -f backend/Dockerfile -t quantum-backend .
docker run -p 8000:8000 --env-file .env -e PORT=8000 quantum-backend
```

**Test (PowerShell):**
```powershell
# Health check
Invoke-WebRequest -Uri http://localhost:8000/health

# Groq mode (fast, ~725ms)
Invoke-WebRequest -Uri http://localhost:8000/query -Method POST -ContentType "application/json" -Body '{"question": "What is a qubit?", "use_groq": true}'

# Custom model (slow, ~50-80s)
Invoke-WebRequest -Uri http://localhost:8000/query -Method POST -ContentType "application/json" -Body '{"question": "What is a qubit?", "use_groq": false}'
```

| Server | Framework | Port | Purpose |
|--------|-----------|------|---------|
| Backend | FastAPI | 8000 | ML model, RAG pipeline, API |
| Frontend | Flask | 3000 | Serves UI, proxies to backend |

---

## Existing Backend Classes

### Retriever (`backend/scripts/retrieval.py`)

Handles semantic search using Voyage AI + Neon pgvector.

```python
class Retriever:
    def embed_query(query: str) -> List[float]
    def search(query: str, top_k: int = 5) -> List[Dict]
    def get_stats() -> Dict
```

### BaseLLM (`backend/scripts/base_inference.py`)

Abstract base class for LLM implementations.

```python
class BaseLLM(ABC):
    @property
    def name(self) -> str
    def generate(context: str, question: str) -> str
    def extract_answer(generated_text: str) -> str
```

### QuantumInference (`backend/scripts/inference.py`)

Handles custom model loading and text generation.

```python
class QuantumInference(BaseLLM):
    def __init__(model_path, tokenizer_path, device)
    def generate(context, question) -> str  # Builds flat prompt
    def extract_answer(generated_text) -> str
    @property
    def name(self) -> str  # Returns "custom"
```

### GroqInference (`backend/scripts/groq_inference.py`)

Handles Groq API calls.

```python
class GroqInference(BaseLLM):
    def __init__()
    def generate(context, question) -> str  # Chat completion API
    def extract_answer(generated_text) -> str
    @property
    def name(self) -> str  # Returns "groq"
```

---

## Architecture

### Two LLM Modes

| Mode | LLM | Speed | Use Case |
|------|-----|-------|----------|
| **Production** | Groq API (Llama 3.3 70B) | ~725ms | Fast UX for users |
| **Demo** | Custom 125.8M | ~50-80s | Prove ML skills to recruiters |

### Pipeline

```
User Question → Voyage AI embed → Neon vector search → Build prompt → LLM generates answer
                                                                       ↓
                                                            use_groq=true  → Groq (~725ms)
                                                            use_groq=false → Custom (~50-80s)
```

### Stack

| Component | Provider | Speed | Cost |
|-----------|----------|-------|------|
| **Frontend** | Flask + Jinja | instant | $0 |
| **Backend** | FastAPI | instant | $0 |
| **Generation (Groq)** | Groq API | ~725ms | $0 (free tier) |
| **Generation (Custom)** | Custom 125.8M | ~50-80s | $0 (lazy loaded) |
| **Embeddings** | Voyage AI | ~100ms | $0 (free tier) |
| **Database** | Neon (pgvector) | ~300ms | $0 (free tier) |
| **Hosting** | Railway | Always on | $5/month |

**Total: ~$5/month**

---

## Topic Scope

**Focus:** Foundational quantum computing concepts for beginners (no math)

| Include | Exclude |
|---------|---------|
| What is a qubit | Dirac notation |
| Superposition (intuitive) | Matrix math |
| Entanglement (intuitive) | Complex numbers |
| Basic gates (what they do, not how) | Gate matrices |
| Quantum vs classical bits | Deep algorithm details |
| Why QC matters (applications overview) | Hardware implementation details |
| Common misconceptions | Bloch sphere math |

**Tone:** Explain like the reader is curious but not technical

---

## Custom Model

### Architecture

| Parameter | Value |
|-----------|-------|
| Type | Decoder-only transformer |
| Parameters | **125,848,320 (125.8M)** |
| Layers | 12 |
| Attention heads | 12 |
| Embedding dimension | 768 |
| Feed-forward dimension | 3072 |
| Vocabulary | 16,384 (custom BPE) |
| Context length | 1024 tokens |

### Key File Locations

| Component | Path |
|-----------|------|
| Model weights | `training/model/final_model.pt` |
| Tokenizer | `training/tokenizer/tokenizer.json` |
| Model architecture | `training/scripts/model.py` |

### Generation Config

| Parameter | Value |
|-----------|-------|
| Temperature | 0.2 |
| Top-k | 30 |
| Pass Rate | 100% |
| Keyword Score | 76-80% |

### Lazy Loading

| Setting | Value |
|---------|-------|
| Load trigger | First request with use_groq=false |
| Unload trigger | 5 min idle |
| Cold start | ~2s |
| Inference | ~50-80s |
| Cost savings | ~$4-5/month |

---

## Groq Integration ✅ COMPLETE

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | llama-3.3-70b-versatile |
| Temperature | 0.2 |
| Max tokens | 300 |
| Response time | ~725ms |

### Test Results (December 27, 2025)

20 quantum computing questions tested, all passed.

---

## RAG System ✅ COMPLETE

### Database Contents

| Source | Count |
|--------|-------|
| claude | 14,400 |
| stackexchange | 10,673 |
| cot | 2,998 |
| **Total** | **28,071** |

### Embedding Configuration

| Parameter | Value |
|-----------|-------|
| Model | voyage-3.5-lite |
| Dimensions | 1024 |
| input_type (documents) | "document" |
| input_type (queries) | "query" |

### Retrieval Quality

| Metric | Value |
|--------|-------|
| Accuracy | **100%** |
| Search time | ~300ms |
| Index type | Exact search |

---

## API Design

### Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/query` | POST | Send question, receive answer |
| `/health` | GET | Health check |

### Request/Response

POST /query:
```json
{
  "question": "What is quantum entanglement?",
  "use_groq": true
}
```

Response:
```json
{
  "answer": "Quantum entanglement is a phenomenon where...",
  "sources": [...],
  "response_time_ms": 725,
  "model_loaded_fresh": false,
  "suggested_question": "What really is Quantum Entanglement and what are its benefits?",
  "llm_used": "groq"
}
```

### Health Response

```json
{
  "status": "ok",
  "model_loaded": false,
  "idle_seconds": null,
  "groq_available": true
}
```

### Suggested Question Feature

Extracts key terms from answer, scores retrieved questions by term matches, filters out questions >60% similar to original. No extra LLM call, no added latency.

---

## UI Design (Flask + Jinja)

### Welcome Screen
- Animated atom icon (CSS keyframes)
- Welcome title and explanation text
- Red disclaimer box: "Free-tier server, 40-90s response time"
- 5 clickable starter questions:
  - "What is a qubit?"
  - "What is superposition?"
  - "What is quantum entanglement?"
  - "What is a quantum gate?"
  - "Why is quantum computing important?"

### Chat Interface
- User messages (red, right-aligned)
- AI messages (gray, left-aligned)
- Response time display
- Suggested follow-up button

### Loading Indicator
- Animated GIF
- Rotating status messages (every ~11s):
  - "Embedding your question with Voyage AI"
  - "Searching knowledge base (28,071 Q&A pairs)"
  - "Retrieved relevant context from Neon database"
  - "Ranking results by relevance"
  - "Building prompt with context"
  - "Loading 125.8M parameter model"
  - "Generating response"
- Patience reminder: "This typically takes 40-90 seconds on our free-tier CPU server."

### Demo Mode UI (Future)

When demo mode is enabled:
- Show warning: "Demo mode uses custom model (~50-80s response time)"
- Show loading indicator during inference
- Display model info: "125.8M parameter transformer trained from scratch"

---

## Hosting & Deployment

### Railway Configuration

| Setting | Value |
|---------|-------|
| Plan | Hobby ($5/month) |
| RAM limit | 8GB |
| Structure | Monorepo (2 services) |
| Always on | Yes |

### Environment Variables

**Backend:**
| Variable | Purpose |
|----------|---------|
| VOYAGE_API_KEY | Embeddings |
| DATABASE_URL | Neon PostgreSQL connection |
| GROQ_API_KEY | Groq API access |

**Frontend:**
| Variable | Purpose |
|----------|---------|
| BACKEND_URL | Backend service URL |

### Deployment Files

**backend/Dockerfile:**
- Python 3.11-slim base
- CPU-only PyTorch
- Git LFS for model download
- Clone repo + pull LFS during build

**frontend/Procfile:**
- gunicorn with --timeout 600

---

## Cost Summary

| Component | Monthly Cost |
|-----------|--------------|
| Railway (hosting) | $5 |
| Groq API | $0 (free tier) |
| Voyage AI | $0 (free tier) |
| Neon | $0 (free tier) |
| **Total** | **~$5/month** |

**Budget:** $12/month
**Actual:** ~$5/month

---

## Cost Protection

| Provider | Hard Cap | Action |
|----------|----------|--------|
| Railway | $10 | Settings > Usage > Hard limit |
| Groq | Free tier | No billing needed |
| Voyage AI | $5 prepaid | Auto-recharge OFF |
| Neon | N/A | Auto-stops at 100 compute hours |

**Maximum exposure: $15/month**

---

## Completed Steps

1. ~~Train custom model~~ ✅ Done (125.8M params)
2. ~~Set up RAG system~~ ✅ Done (100% accuracy)
3. ~~Tune generation params~~ ✅ Done (temp=0.2, top_k=30)
4. ~~Create backend classes~~ ✅ Done (Retriever, BaseLLM, Inference classes)
5. ~~Create FastAPI app~~ ✅ Done (lazy loading, suggested questions)
6. ~~Build frontend~~ ✅ Done (Flask + Jinja)
7. ~~Deploy to Railway~~ ✅ Done (live)
8. ~~Add Groq integration~~ ✅ Done (tested locally)

## Future Improvements

9. **Deploy Groq to Railway** ⬜ Pending (add GROQ_API_KEY)
10. **Add frontend toggle for LLM mode** ⬜ Pending
11. **Implement response caching** ⬜ Pending

---

*Document version: 20.0*
*Last updated: December 27, 2025*
