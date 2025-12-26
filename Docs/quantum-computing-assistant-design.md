# Quantum Computing Assistant - Design Document

## Overview

Design specification for a web-based application that answers questions about foundational quantum computing concepts. Powered by custom-trained 125.8M parameter transformer, with Groq API as optional fast mode (to be added later).

**Related Documents:**
- Infrastructure Planning: `initial-exploratory-brainstorming.md`
- Implementation Plan: `implementation-plan.md`
- Model Investigation Report: `model_investigation_report.md`

**Disclaimer:** Model weights are private and not distributed. Training data used for educational/personal purposes only.

---

## Project Goals

**Primary Goal:** Portfolio demonstration for recruiters

This project proves ability to:
- Design transformer architecture from scratch
- Build complete two-phase training pipeline
- Train on real HPC hardware (H100 GPUs)
- Deploy end-to-end ML system with RAG

**Target Audience:** Recruiters evaluating ML skills, students curious about quantum computing

---

## Current Status

| Component | Status |
|-----------|--------|
| Custom Model (v5) | ✅ COMPLETE (125.8M params, 100% pass rate) |
| RAG System | ✅ COMPLETE (100% retrieval) |
| Parameter Tuning | ✅ COMPLETE (temp=0.2, top_k=30) |
| Backend Classes | ✅ COMPLETE (Retriever, QuantumInference, Pipeline) |
| FastAPI App | ✅ COMPLETE (lazy loading, suggested questions) |
| Frontend | ✅ COMPLETE (Flask + Jinja) |
| Deployment | ⬜ Pending |

**Next Action:** Deploy to Railway (Phase 5)

---

## Project Structure

```
Quantum-Computing-LLM/
├── Docs/
│   └── *.md
│
├── training/
│   ├── model/
│   │   ├── final_model.pt              # 125.8M params
│   │   └── config.json
│   ├── tokenizer/
│   │   └── tokenizer.json              # 16K vocab BPE
│   └── scripts/
│       └── model.py                    # QuantumLLM architecture
│
├── backend/
│   ├── scripts/                        # ✅ Existing utilities
│   │   ├── retrieval.py                # Retriever class
│   │   ├── inference.py                # QuantumInference class
│   │   └── pipeline.py                 # QuantumRAGPipeline class
│   └── app/                            # ✅ FastAPI app
│       ├── __init__.py
│       ├── config.py                   # Environment variables
│       └── main.py                     # Endpoints, lazy loading
│
├── frontend/                           # ✅ Flask app
│   ├── app.py                          # Flask server (port 3000)
│   ├── requirements.txt                # flask, requests
│   ├── static/
│   │   └── style.css                   # All styles
│   └── templates/
│       └── index.html                  # Jinja template with JS
│
└── .env                                # API keys
```

---

## Run Commands

**Terminal 1: Backend (FastAPI)**
```powershell
cd E:\Personal_projects\Quantum-Computing-LLM
.\venv\Scripts\Activate
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2: Frontend (Flask)**
```powershell
cd E:\Personal_projects\Quantum-Computing-LLM
.\venv\Scripts\Activate
cd frontend
python app.py
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

### QuantumInference (`backend/scripts/inference.py`)

Handles model loading and text generation.

```python
class QuantumInference:
    def __init__(model_path, tokenizer_path, device)
    def generate(prompt, max_new_tokens=150, temperature=0.2, top_k=30) -> str
    def extract_answer(generated_text) -> str
```

### QuantumRAGPipeline (`backend/scripts/pipeline.py`)

Combines retrieval and inference into a single query interface.

```python
class QuantumRAGPipeline:
    def __init__(model_path, tokenizer_path, device)
    def query(question, top_k_retrieval=5, ...) -> Dict
        # Returns: answer, sources, suggested_questions
```

---

## Architecture

### Two LLM Modes (Implementation Order)

| Mode | LLM | Speed | Status |
|------|-----|-------|--------|
| **Custom** | Custom 125.8M | ~35-37s | ✅ Implemented |
| **Production** | Groq API | ~1-2s | ⬜ Add later |

### Pipeline

```
User Question → Voyage AI embed → Neon vector search → Build prompt → LLM generates answer
                                                                         ↓
                                                              Custom model (implemented)
                                                              Groq API (later)
```

### Stack

| Component | Provider | Speed | Cost |
|-----------|----------|-------|------|
| **Frontend** | Flask + Jinja | instant | $0 |
| **Backend** | FastAPI | instant | $0 |
| **Generation (Custom)** | Custom 125.8M | ~35-37s | $0 (lazy loaded) |
| **Generation (Groq)** | Groq API | ~1-2s | $0 (free tier) |
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
| Load trigger | First request |
| Unload trigger | 5 min idle |
| Cold start | ~5s |
| Inference | ~35-37s |
| Cost savings | ~$4-5/month |

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
  "question": "What is quantum entanglement?"
}
```

Response:
```json
{
  "answer": "Quantum entanglement is a phenomenon where...",
  "response_time_ms": 36000,
  "model_loaded_fresh": false,
  "suggested_question": "What really is Quantum Entanglement and what are its benefits?"
}
```

### Suggested Question Feature

Extracts key terms from answer, scores retrieved questions by term matches, filters out questions >60% similar to original. No extra LLM call, no added latency.

---

## UI Design (Flask + Jinja)

### Welcome Screen
- Animated atom icon (CSS keyframes)
- Welcome title and explanation text
- Yellow disclaimer box: "Free-tier server, 40-90s response time"
- 5 clickable starter questions:
  - "What is a qubit?"
  - "What is superposition?"
  - "What is quantum entanglement?"
  - "What is a quantum gate?"
  - "Why is quantum computing important?"

### Chat Interface
- User messages (blue, right-aligned)
- AI messages (gray, left-aligned)
- Response time display
- Suggested follow-up button

### Loading Indicator
- Animated atom with orbiting electrons
- Rotating status messages (every 3s):
  - "Searching knowledge base..."
  - "Retrieving relevant context..."
  - "Processing with 125.8M parameters..."
  - etc.
- Quantum facts (every 8s)
- Patience reminder: "40-90 seconds on free-tier server"

### Demo Mode UI (Later)

When demo mode is enabled:
- Show warning: "Demo mode uses custom model (~35s response time)"
- Show loading indicator during inference
- Display model info: "125.8M parameter transformer trained from scratch"

---

## Hosting & Deployment

### Railway Configuration

| Setting | Value |
|---------|-------|
| Plan | Hobby ($5/month) |
| RAM limit | 8GB |
| Structure | Monorepo |
| Always on | Yes |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| VOYAGE_API_KEY | Embeddings |
| DATABASE_URL | Neon PostgreSQL connection |
| GROQ_API_KEY | Groq generation (later) |

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

## Next Steps

1. ~~Train custom model~~ ✅ Done (125.8M params)
2. ~~Set up RAG system~~ ✅ Done (100% accuracy)
3. ~~Tune generation params~~ ✅ Done (temp=0.2, top_k=30)
4. ~~Create backend classes~~ ✅ Done (Retriever, Inference, Pipeline)
5. ~~Create FastAPI app~~ ✅ Done (lazy loading, suggested questions)
6. ~~Build frontend~~ ✅ Done (Flask + Jinja)
7. **Deploy to Railway** ⬜ Pending
8. **Add Groq integration** ⬜ Pending (later)
9. **Add demo mode toggle** ⬜ Pending (later)

---

*Document version: 18.0*
*Last updated: December 26, 2025*
