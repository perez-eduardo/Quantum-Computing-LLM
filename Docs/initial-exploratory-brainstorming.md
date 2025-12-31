# Initial Exploratory Brainstorming: Next LLM Project Stack

**Date:** December 20, 2025
**Last Updated:** December 31, 2025
**Project:** Quantum Computing LLM
**Purpose:** Portfolio demonstration piece
**Expected Traffic:** Minimal (recruiters, students)

---

## Problem Statement

The Philippine Legal Assistant project revealed infrastructure limitations:

1. **Groq free tier:** Shared limits across all users caused rate limiting
2. **Render free tier:** 512MB RAM cannot handle local Sentence Transformers model

Need a stack that is:
- Cost-optimized ($12/month budget)
- Reliable (no shared rate limits)
- Fast (minimal cold starts)

---

## Final Architecture

### Two LLM Modes

| Mode | LLM | Host | Speed | Status |
|------|-----|------|-------|--------|
| Production | Groq API (Llama 3.3 70B) | Groq Cloud | ~2-3s | ✅ Complete |
| Demo | Custom 140M | Modal (T4 GPU) | ~35-60s | ✅ Complete |

### Stack

| Component | Provider | Cost | Notes |
|-----------|----------|------|-------|
| **Frontend** | Flask + Jinja | $0 | Port 3000, Railway |
| **Backend** | FastAPI | $0 | Port 8000, Railway |
| **Hosting** | Railway (Hobby) | $5/month | Monorepo, always on |
| **LLM (Custom)** | Custom 140M | $0 | Modal T4 GPU, ~35-60s |
| **LLM (Production)** | Groq API | $0 | Free tier, ~2-3s |
| **Embeddings** | Voyage AI | $0 | 200M free tokens |
| **Database** | Neon (free) | $0 | PostgreSQL + pgvector |

**Total: ~$5/month** (within $12 budget)

---

## Live URLs

| Service | URL |
|---------|-----|
| Frontend | https://quantum-computing-llm.up.railway.app |
| Backend | https://quantum-computing-llm-backend.up.railway.app |
| API Docs | https://quantum-computing-llm-backend.up.railway.app/docs |
| Health Check | https://quantum-computing-llm-backend.up.railway.app/health |
| Modal Query | https://perez-eduardo--quantum-llm-query.modal.run |
| Modal Health | https://perez-eduardo--quantum-llm-health.modal.run |

---

## Project Structure

```
Quantum-Computing-LLM/
├── Docs/
│   └── *.md
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
│   ├── Dockerfile                      # CPU PyTorch, Git LFS
│   ├── Procfile
│   ├── requirements.txt                # groq>=0.11.0
│   ├── scripts/
│   │   ├── retrieval.py                # Retriever class
│   │   ├── base_inference.py           # BaseLLM abstract class
│   │   ├── groq_inference.py           # GroqInference(BaseLLM)
│   │   └── modal_inference.py          # ModalInference (calls Modal API)
│   └── app/
│       ├── __init__.py
│       ├── config.py                   # Environment variables + Groq + Modal
│       └── main.py                     # Endpoints, LLM selection
│
├── frontend/
│   ├── Dockerfile                      # gunicorn --timeout 300
│   ├── Procfile
│   ├── app.py                          # Flask server
│   ├── requirements.txt
│   ├── static/
│   │   ├── css/style.css
│   │   ├── js/main.js                  # Model selector, health checks, retry
│   │   └── images/
│   └── templates/
│       └── index.html                  # Jinja template with modals
│
├── modal/
│   ├── inference.py                    # Modal app with QuantumLLM
│   ├── test_local.py                   # Local testing script
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

# Groq (fast)
Invoke-WebRequest -Uri http://localhost:8000/query -Method POST -ContentType "application/json" -Body '{"question": "What is a qubit?", "model": "groq"}'

# Custom via Modal (slower)
Invoke-WebRequest -Uri http://localhost:8000/query -Method POST -ContentType "application/json" -Body '{"question": "What is a qubit?", "model": "custom"}'
```

---

## Existing Backend Classes

### Retriever (`backend/scripts/retrieval.py`)

```python
class Retriever:
    def embed_query(query) -> List[float]   # Voyage AI
    def search(query, top_k) -> List[Dict]  # Neon pgvector
    def get_stats() -> Dict
```

### BaseLLM (`backend/scripts/base_inference.py`)

```python
class BaseLLM(ABC):
    @property
    def name(self) -> str
    def generate(context, question) -> str
    def extract_answer(generated_text) -> str
```

### GroqInference (`backend/scripts/groq_inference.py`)

```python
class GroqInference(BaseLLM):
    def __init__()
    def generate(context, question) -> str  # Chat completion API
    def extract_answer(generated_text) -> str
    @property
    def name(self) -> str  # Returns "groq"
```

### ModalInference (`backend/scripts/modal_inference.py`)

```python
class ModalInference(BaseLLM):
    def __init__()
    def generate(context, question) -> str  # Calls Modal API
    def extract_answer(generated_text) -> str
    @property
    def name(self) -> str  # Returns "custom"
```

---

## Pipeline

```
User Question → Voyage AI embed → Neon vector search → Build prompt → LLM generates answer
                                                                       ↓
                                                        model="groq"   → Groq (~2-3s)
                                                        model="custom" → Modal (~35-60s)
```

---

## Custom Model

### Why Keep It?

Even though Groq handles production, the custom model demonstrates:
- Transformer architecture design
- Two-phase training pipeline
- HPC cluster experience (H100 GPUs)
- Data processing and cleaning
- Modal serverless GPU deployment

Recruiters can toggle "Custom" to see the trained model in action.

### Key File Locations

| Component | Path |
|-----------|------|
| Model weights | `training/model/final_model.pt` |
| Tokenizer | `training/tokenizer/tokenizer.json` |
| Model architecture | `training/scripts/model.py` |
| Modal deployment | `modal/inference.py` |

### Configuration

| Parameter | Value |
|-----------|-------|
| Parameters | 140,004,480 (140M) |
| Temperature | 0.2 |
| Top-k | 30 |
| Pass Rate | 100% |
| Inference time | ~35-60s (Modal T4 GPU) |

### Parameter Tuning (December 26, 2025)

Battery test on HPC: 24 combinations × 20 questions = 480 tests in 5.8 minutes.

| Parameters | Pass Rate | Keyword Score |
|------------|-----------|---------------|
| **temp=0.2, top_k=30** | **100%** | **80.5%** |
| temp=0.4, top_k=20 | 100% | 78.8% |
| temp=0.3, top_k=50 (old) | 100% | 74.2% |

---

## Groq Integration ✅ COMPLETE

### Configuration

| Setting | Value |
|---------|-------|
| Model | llama-3.3-70b-versatile |
| Temperature | 0.2 |
| Max tokens | 300 |
| Response time | ~2-3s |

### Test Results (December 27, 2025)

20 quantum computing questions tested, all passed.

---

## Modal Deployment ✅ COMPLETE

### Configuration

| Setting | Value |
|---------|-------|
| GPU | T4 (16GB VRAM) |
| Free tier | $30/month |
| Cold start | 3-4 seconds |
| Inference | ~35-60 seconds total |
| Volume | quantum-model-volume |

---

## RAG System ✅ COMPLETE

### Retrieval Quality: 100%

| Metric | Value |
|--------|-------|
| Q&A embeddings | 28,071 |
| Test pass rate | **100%** |
| Search time | ~300ms |

### Index Fix (December 25, 2025)

IVFFlat approximate index was missing exact matches. Removed for exact search.

---

## Frontend Features ✅ COMPLETE

### Model Selector
- Dropdown in header (Groq vs Custom)
- Toast notification when Custom selected

### Health Check System
- Pre-query health check on every request
- Wake-up indicator for cold starts
- Specific error after 10 failed attempts

### Loading Indicators
- Groq: "Thinking..."
- Custom: Pipeline animation with stages

### Error Handling
- Retry button on failed queries
- Timeout handling (30s Groq, 180s Custom)

---

## Component Details

### Generation: Groq API ✅

| Setting | Value |
|---------|-------|
| Model | llama-3.3-70b-versatile |
| Free tier | 30 req/min, 14,400 req/day |
| Speed | ~2-3s per response |

### Generation: Custom Model (Modal) ✅

| Setting | Value |
|---------|-------|
| Parameters | 140M |
| GPU | T4 (Modal) |
| Speed | ~35-60s per response |
| temp | 0.2 |
| top_k | 30 |

### Embeddings: Voyage AI

| Provider | Free Tokens | Free RPM |
|----------|-------------|----------|
| Voyage AI | 200M | 2000 RPM |

### Database: Neon Free Tier

| Resource | Limit | Current Usage |
|----------|-------|---------------|
| Storage | 0.5 GB | 28,071 embeddings |

### Hosting: Railway Hobby

| Setting | Value |
|---------|-------|
| Cost | $5/month |
| RAM limit | 8GB |
| Always on | Yes |

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
**Headroom:** ~$7/month

---

## Deployment Issues Resolved

| Issue | Cause | Solution |
|-------|-------|----------|
| Model file 130 bytes | Git LFS pointer not pulled | Clone repo + `git lfs pull` in Dockerfile |
| Tokenizer error | Version mismatch (0.15.0 vs 0.22.1) | Updated requirements.txt |
| 502 errors | Gunicorn 30s default timeout | Set `--timeout 300` in Dockerfile |
| Silent crashes | No logging | Added print statements |
| Groq proxy error | groq==0.4.2 httpx issue | Updated to groq>=0.11.0 |
| Modal tokenizer error | Wrong tokenizers version | Set tokenizers==0.22.1 in Modal |
| Frontend 502 on cold start | Backend not ready | Added health check before every query |

---

## Lessons Learned

1. **Don't trust synthetic data blindly.** ChatGPT generated 94% garbage.

2. **Two-phase training works.** Books for prose, context Q&A for RAG usage.

3. **140M params is the sweet spot.** 1.2M = gibberish, 140M = coherent.

4. **Lower temperature = better.** temp=0.2, top_k=30 achieved 100% pass rate.

5. **Modal for GPU inference.** Much faster than Railway CPU (~35-60s vs ~50-80s).

6. **IVFFlat index can miss results.** Exact search more reliable.

7. **Groq solves the speed problem.** ~2-3s vs ~35-60s for custom model.

8. **Extraction function matters.** rfind() grabbed wrong answer, find() fixed it.

9. **HPC speeds up testing.** 480 tests in 5.8 min vs ~280 min on CPU.

10. **Build backend classes first.** Retriever, Inference classes exist and work.

11. **Flask + Jinja is simpler than React.** Single Python file, no npm/node needed.

12. **Git LFS needs explicit pull in Docker.** Railway doesn't auto-pull LFS files.

13. **Tokenizers version must match training.** Caused cryptic deserialization errors.

14. **Gunicorn timeout must exceed response time.** 30s default too short for ML inference.

15. **Abstract base class pattern works.** Clean separation between Groq and Modal.

16. **groq package version matters.** 0.4.2 incompatible, 0.11.0+ works.

17. **Use --env-file for Docker.** Cleaner than passing -e for each variable.

18. **Health check before every query.** Prevents 502 from sleeping servers.

19. **Dockerfile timeout must match Procfile.** Both need 300s for custom model.

20. **Model config inside .pt file.** QuantumLLM.load() extracts config automatically.

---

*Document version: 18.0*
