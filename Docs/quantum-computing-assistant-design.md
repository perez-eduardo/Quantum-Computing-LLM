# Quantum Computing Assistant - Design Document

**Last Updated:** December 31, 2025

## Overview

Design specification for a web-based application that answers questions about foundational quantum computing concepts. Powered by custom-trained 140M parameter transformer (demo mode via Modal) and Groq API with Llama 3.3 70B (production mode).

**Related Documents:**
- Infrastructure Planning: `initial-exploratory-brainstorming.md`
- Implementation Plan: `implementation-plan.md`
- Model Investigation Report: `model_investigation_report.md`
- Groq Integration Plan: `groq-integration-plan.md`
- Modal Deployment Plan: `modal-deployment-plan.md`

**Disclaimer:** Model weights are private and not distributed. Training data used for educational/personal purposes only.

---

## Project Goals

**Primary Goal:** Portfolio demonstration for recruiters

This project proves ability to:
- Design transformer architecture from scratch
- Build complete two-phase training pipeline
- Train on real HPC hardware (H100 GPUs)
- Deploy end-to-end ML system with RAG
- Integrate third-party APIs (Groq, Voyage AI, Modal)
- Build production-ready frontend with health checks and error handling

**Target Audience:** Recruiters evaluating ML skills, students curious about quantum computing

---

## Current Status

| Component | Status |
|-----------|--------|
| Custom Model (v5) | ✅ COMPLETE (140M params, 100% pass rate) |
| RAG System | ✅ COMPLETE (100% retrieval) |
| Parameter Tuning | ✅ COMPLETE (temp=0.2, top_k=30) |
| Backend Classes | ✅ COMPLETE (Retriever, BaseLLM, GroqInference, ModalInference) |
| FastAPI App | ✅ COMPLETE (dual-mode, suggested questions) |
| Frontend | ✅ COMPLETE (Flask + Jinja, model selector, health checks) |
| Deployment | ✅ COMPLETE (Railway, live) |
| Groq Integration | ✅ COMPLETE (deployed, ~2-3s) |
| Modal Deployment | ✅ COMPLETE (T4 GPU, ~35-60s) |

---

## Live URLs

| Service | URL |
|---------|-----|
| **Frontend** | https://quantum-computing-llm.up.railway.app |
| **Backend** | https://quantum-computing-llm-backend.up.railway.app |
| **API Docs** | https://quantum-computing-llm-backend.up.railway.app/docs |
| **Health Check** | https://quantum-computing-llm-backend.up.railway.app/health |
| **Modal Query** | https://perez-eduardo--quantum-llm-query.modal.run |
| **Modal Health** | https://perez-eduardo--quantum-llm-health.modal.run |

---

## Project Structure

```
Quantum-Computing-LLM/
├── Docs/
│   ├── implementation-plan.md
│   ├── initial-exploratory-brainstorming.md
│   ├── model_investigation_report.md
│   ├── quantum-computing-assistant-design.md
│   ├── groq-integration-plan.md
│   └── modal-deployment-plan.md
│
├── training/
│   ├── model/
│   │   ├── final_model.pt              # 140M params (510MB, Git LFS)
│   │   └── config.json
│   ├── tokenizer/
│   │   └── tokenizer.json              # 16K vocab BPE
│   └── scripts/
│       └── model.py                    # QuantumLLM architecture
│
├── backend/
│   ├── Dockerfile
│   ├── Procfile
│   ├── requirements.txt                # tokenizers==0.22.1, groq>=0.11.0
│   ├── scripts/
│   │   ├── retrieval.py                # Retriever class
│   │   ├── base_inference.py           # BaseLLM abstract class
│   │   ├── groq_inference.py           # GroqInference(BaseLLM)
│   │   └── modal_inference.py          # ModalInference(BaseLLM)
│   └── app/
│       ├── __init__.py
│       ├── config.py                   # Environment variables
│       └── main.py                     # Endpoints, LLM selection
│
├── frontend/
│   ├── Dockerfile                      # gunicorn --timeout 300
│   ├── Procfile
│   ├── app.py                          # Flask server
│   ├── requirements.txt
│   ├── static/
│   │   ├── css/style.css
│   │   ├── js/main.js                  # Model selector, health checks
│   │   └── images/
│   └── templates/
│       └── index.html                  # Jinja template with modals
│
├── modal/
│   ├── inference.py                    # Modal app with QuantumLLM
│   ├── test_local.py                   # Local testing
│   └── venv/                           # Modal CLI venv
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

# Groq mode (fast, ~2-3s)
Invoke-WebRequest -Uri http://localhost:8000/query -Method POST -ContentType "application/json" -Body '{"question": "What is a qubit?", "model": "groq"}'

# Custom model via Modal (~35-60s)
Invoke-WebRequest -Uri http://localhost:8000/query -Method POST -ContentType "application/json" -Body '{"question": "What is a qubit?", "model": "custom"}'
```

| Server | Framework | Port | Purpose |
|--------|-----------|------|---------|
| Backend | FastAPI | 8000 | RAG pipeline, LLM routing |
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

### GroqInference (`backend/scripts/groq_inference.py`)

Handles Groq API calls.

```python
class GroqInference(BaseLLM):
    def __init__()
    def generate(context, question) -> str
    def extract_answer(generated_text) -> str
    @property
    def name(self) -> str  # Returns "groq"
```

### ModalInference (`backend/scripts/modal_inference.py`)

Handles Modal API calls for custom model.

```python
class ModalInference(BaseLLM):
    def __init__()
    def generate(context, question) -> str  # Calls Modal endpoint
    def extract_answer(generated_text) -> str
    @property
    def name(self) -> str  # Returns "custom"
```

---

## Architecture

### Two LLM Modes

| Mode | LLM | Host | Speed | Use Case |
|------|-----|------|-------|----------|
| **Production** | Groq API (Llama 3.3 70B) | Groq Cloud | ~2-3s | Fast UX for users |
| **Demo** | Custom 140M | Modal (T4 GPU) | ~35-60s | Prove ML skills to recruiters |

### Pipeline

```
User Question → Railway Frontend → Railway Backend → Voyage AI embed → Neon vector search
                                                                              ↓
                                                         model="groq"   → Groq API (~2-3s)
                                                         model="custom" → Modal API (~35-60s)
```

### Stack

| Component | Provider | Speed | Cost |
|-----------|----------|-------|------|
| **Frontend** | Flask + Jinja | instant | $0 |
| **Backend** | FastAPI | instant | $0 |
| **Generation (Groq)** | Groq API | ~2-3s | $0 (free tier) |
| **Generation (Custom)** | Modal (T4 GPU) | ~35-60s | $0 (free tier) |
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
| Parameters | **140,004,480 (140M)** |
| Layers | 12 |
| Attention heads | 12 |
| Embedding dimension | 768 |
| Feed-forward dimension | 3072 |
| Vocabulary | 16,384 (custom BPE) |
| Context length | 1024 tokens |

**Important:** Model config is stored INSIDE the .pt file. Use `QuantumLLM.load(path, device)` to load.

### Key File Locations

| Component | Path |
|-----------|------|
| Model weights | `training/model/final_model.pt` |
| Tokenizer | `training/tokenizer/tokenizer.json` |
| Model architecture | `training/scripts/model.py` |
| Modal deployment | `modal/inference.py` |

### Generation Config

| Parameter | Value |
|-----------|-------|
| Temperature | 0.2 |
| Top-k | 30 |
| Pass Rate | 100% |

### Modal Deployment

| Setting | Value |
|---------|-------|
| GPU | T4 (16GB VRAM) |
| Cold start | 3-4 seconds |
| Inference | ~35-60 seconds total |
| Free tier | $30/month |

---

## Groq Integration ✅ COMPLETE

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | llama-3.3-70b-versatile |
| Temperature | 0.2 |
| Max tokens | 300 |
| Response time | ~2-3s |

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
  "model": "groq"
}
```

Response:
```json
{
  "answer": "Quantum entanglement is a phenomenon where...",
  "sources": [...],
  "response_time_ms": 2500,
  "suggested_question": "What really is Quantum Entanglement and what are its benefits?",
  "model_used": "groq"
}
```

### Health Response

```json
{
  "status": "ok",
  "groq_available": true,
  "modal_available": true
}
```

### Suggested Question Feature

Extracts key terms from answer, scores retrieved questions by term matches, filters out questions >60% similar to original. No extra LLM call, no added latency.

---

## UI Design (Flask + Jinja)

### Model Selector
- Dropdown in header (top-left)
- Options: Groq (default), Custom
- Toast notification when Custom selected
- Displays response time expectations

### Welcome Screen
- Animated atom icon (CSS keyframes)
- Welcome title and explanation text
- Cold start disclaimer
- 4 clickable starter questions:
  - "What is a qubit?"
  - "What is superposition?"
  - "What is quantum entanglement?"
  - "What is a quantum gate?"

### Chat Interface
- User messages (blue, right-aligned)
- AI messages (gray, left-aligned)
- Response time and model display
- Suggested follow-up button

### Health Check System
- Initial startup check (waits for backend)
- Pre-query health check on EVERY request
- Wake-up indicator for cold starts
- "Connecting to backend..." for active sessions
- Specific error after 10 failed attempts

### Loading Indicators
- Groq: "Thinking..."
- Custom: Pipeline animation with 7 stages:
  - "Embedding your question with Voyage AI"
  - "Searching knowledge base (28,071 Q&A pairs)"
  - "Retrieved relevant context from Neon database"
  - "Ranking results by relevance"
  - "Building prompt with context"
  - "Loading 140M parameter model"
  - "Generating response"

### Error Handling
- Retry button on failed queries
- Timeout handling (30s Groq, 180s Custom)
- "Backend server did not respond after 10 attempts..." message

### Modals
- **Models modal:** Explains Groq vs Custom, shows specs
- **About modal:** Project details, tech stack, training info, contact links

---

## Hosting & Deployment

### Railway Configuration

| Setting | Value |
|---------|-------|
| Plan | Hobby ($5/month) |
| RAM limit | 8GB |
| Structure | Monorepo (2 services) |
| Always on | Yes |

### Modal Configuration

| Setting | Value |
|---------|-------|
| GPU | T4 (16GB VRAM) |
| Free tier | $30/month |
| Idle timeout | 5 minutes |
| Volume | quantum-model-volume |

### Environment Variables

**Backend (Railway):**
| Variable | Purpose |
|----------|---------|
| VOYAGE_API_KEY | Embeddings |
| DATABASE_URL | Neon PostgreSQL connection |
| GROQ_API_KEY | Groq API access |
| MODAL_URL | Modal endpoint URL |

**Frontend (Railway):**
| Variable | Purpose |
|----------|---------|
| BACKEND_URL | Backend service URL |

### Deployment Files

**backend/Dockerfile:**
- Python 3.11-slim base
- CPU-only PyTorch
- Git LFS for model download

**frontend/Dockerfile:**
- Python 3.11-slim base
- gunicorn with --timeout 300

---

## Cost Summary

| Component | Monthly Cost |
|-----------|--------------|
| Railway (hosting) | $5 |
| Groq API | $0 (free tier) |
| Modal | $0 (free tier, $30 credit) |
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
| Modal | $30 | Free tier limit |
| Groq | Free tier | No billing needed |
| Voyage AI | $5 prepaid | Auto-recharge OFF |
| Neon | N/A | Auto-stops at 100 compute hours |

**Maximum exposure: ~$15/month**

---

## All Steps Completed ✅

1. ~~Train custom model~~ ✅ Done (140M params)
2. ~~Set up RAG system~~ ✅ Done (100% accuracy)
3. ~~Tune generation params~~ ✅ Done (temp=0.2, top_k=30)
4. ~~Create backend classes~~ ✅ Done (Retriever, BaseLLM, Inference classes)
5. ~~Create FastAPI app~~ ✅ Done (dual-mode, suggested questions)
6. ~~Build frontend~~ ✅ Done (Flask + Jinja)
7. ~~Deploy to Railway~~ ✅ Done (live)
8. ~~Add Groq integration~~ ✅ Done (deployed)
9. ~~Deploy custom model to Modal~~ ✅ Done (T4 GPU)
10. ~~Add frontend model selector~~ ✅ Done
11. ~~Add health check before every query~~ ✅ Done
12. ~~Add retry button~~ ✅ Done

---

## Future Improvements (Optional)

- Response caching for common questions
- Rate limiting for abuse prevention
- Analytics dashboard

---

*Document version: 21.0*
