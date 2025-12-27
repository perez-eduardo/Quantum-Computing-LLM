# Initial Exploratory Brainstorming: Next LLM Project Stack

**Date:** December 20, 2025 (Updated December 27, 2025)
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

| Mode | LLM | Speed | Status |
|------|-----|-------|--------|
| Production | Groq API (Llama 3.3 70B) | ~725ms | ✅ Complete |
| Demo | Custom 125.8M | ~50-80s | ✅ Deployed |

### Stack

| Component | Provider | Cost | Notes |
|-----------|----------|------|-------|
| **Frontend** | Flask + Jinja | $0 | Port 3000 |
| **Backend** | FastAPI | $0 | Port 8000 |
| **Hosting** | Railway (Hobby) | $5/month | Monorepo, always on |
| **LLM (Custom)** | Custom 125.8M | $0 | Lazy loaded, ~50-80s |
| **LLM (Production)** | Groq API | $0 | Free tier, ~725ms |
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

---

## Project Structure

```
Quantum-Computing-LLM/
├── Docs/
│   └── *.md
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
│   ├── Dockerfile                      # CPU PyTorch, Git LFS
│   ├── Procfile
│   ├── requirements.txt                # groq>=0.11.0
│   ├── scripts/
│   │   ├── retrieval.py                # Retriever class
│   │   ├── base_inference.py           # BaseLLM abstract class
│   │   ├── inference.py                # QuantumInference(BaseLLM)
│   │   └── groq_inference.py           # GroqInference(BaseLLM)
│   └── app/
│       ├── __init__.py
│       ├── config.py                   # Environment variables + Groq
│       └── main.py                     # Endpoints, LLM selection
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

# Groq (fast)
Invoke-WebRequest -Uri http://localhost:8000/query -Method POST -ContentType "application/json" -Body '{"question": "What is a qubit?", "use_groq": true}'

# Custom (slow)
Invoke-WebRequest -Uri http://localhost:8000/query -Method POST -ContentType "application/json" -Body '{"question": "What is a qubit?", "use_groq": false}'
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

### QuantumInference (`backend/scripts/inference.py`)

```python
class QuantumInference(BaseLLM):
    def __init__(model_path, tokenizer_path, device)
    def generate(context, question) -> str  # Flat prompt format
    def extract_answer(generated_text) -> str
    @property
    def name(self) -> str  # Returns "custom"
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

---

## Pipeline

```
User Question → Voyage AI embed → Neon vector search → Build prompt → LLM generates answer
                                                                       ↓
                                                            use_groq=true  → Groq (~725ms)
                                                            use_groq=false → Custom (~50-80s)
```

---

## Custom Model

### Why Keep It?

Even though Groq handles production, the custom model demonstrates:
- Transformer architecture design
- Two-phase training pipeline
- HPC cluster experience (H100 GPUs)
- Data processing and cleaning

Recruiters can toggle "Demo Mode" to see the trained model in action.

### Key File Locations

| Component | Path |
|-----------|------|
| Model weights | `training/model/final_model.pt` |
| Tokenizer | `training/tokenizer/tokenizer.json` |
| Model architecture | `training/scripts/model.py` |

### Configuration

| Parameter | Value |
|-----------|-------|
| Temperature | 0.2 |
| Top-k | 30 |
| Pass Rate | 100% |
| Keyword Score | 76-80% |
| Inference time | ~50-80s (Railway CPU) |

### Parameter Tuning (December 26, 2025)

Battery test on HPC: 24 combinations × 20 questions = 480 tests in 5.8 minutes.

| Parameters | Pass Rate | Keyword Score |
|------------|-----------|---------------|
| **temp=0.2, top_k=30** | **100%** | **80.5%** |
| temp=0.4, top_k=20 | 100% | 78.8% |
| temp=0.3, top_k=50 (old) | 100% | 74.2% |

### Lazy Loading Strategy

| Approach | RAM Usage | Cost |
|----------|-----------|------|
| Always loaded | ~700MB constant | ~$6-8/month |
| **Lazy load + timeout** | ~200MB idle | **~$2-3/month** |

Model loads on first request, unloads after 5 min idle.

---

## Groq Integration ✅ COMPLETE

### Configuration

| Setting | Value |
|---------|-------|
| Model | llama-3.3-70b-versatile |
| Temperature | 0.2 |
| Max tokens | 300 |
| Response time | ~725ms |

### Test Results (December 27, 2025)

20 quantum computing questions tested, all passed.

Sample response:
```json
{
  "answer": "A qubit, short for quantum bit, is the basic unit of information...",
  "sources": [...],
  "response_time_ms": 725,
  "model_loaded_fresh": false,
  "suggested_question": "can you explain qubits in simple terms?",
  "llm_used": "groq"
}
```

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

## Component Details

### Generation: Custom Model

| Setting | Value |
|---------|-------|
| Parameters | 125.8M |
| Speed | ~50-80s per response |
| temp | 0.2 |
| top_k | 30 |

### Generation: Groq API ✅

| Setting | Value |
|---------|-------|
| Model | llama-3.3-70b-versatile |
| Free tier | 30 req/min, 14,400 req/day |
| Speed | ~725ms per response |

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

## Frontend Features (Flask + Jinja)

### Welcome Screen
- Animated atom icon
- 5 starter questions
- Free-tier disclaimer (40-90s warning)

### Chat Interface
- User messages (blue, right-aligned)
- AI messages (gray, left-aligned)
- Response time display
- Suggested follow-up button

### Loading Indicator
- Animated GIF
- Rotating status messages (every 3s)
- Patience reminder

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
**Headroom:** ~$7/month

---

## Deployment Issues Resolved (December 27, 2025)

| Issue | Cause | Solution |
|-------|-------|----------|
| Model file 130 bytes | Git LFS pointer not pulled | Clone repo + `git lfs pull` in Dockerfile |
| Tokenizer error | Version mismatch (0.15.0 vs 0.22.1) | Updated requirements.txt |
| 502 errors | Gunicorn 30s default timeout | Set `--timeout 600` in Procfile |
| Silent crashes | No logging | Added print statements (later removed) |
| Groq proxy error | groq==0.4.2 httpx issue | Updated to groq>=0.11.0 |

---

## Lessons Learned

1. **Don't trust synthetic data blindly.** ChatGPT generated 94% garbage.

2. **Two-phase training works.** Books for prose, context Q&A for RAG usage.

3. **125M params is the sweet spot.** 1.2M = gibberish, 125M = coherent.

4. **Lower temperature = better.** temp=0.2, top_k=30 achieved 100% pass rate.

5. **Lazy loading saves cost.** $2-3/month vs $6-8/month always loaded.

6. **IVFFlat index can miss results.** Exact search more reliable.

7. **Groq solves the speed problem.** ~725ms vs ~50-80s for custom model.

8. **Extraction function matters.** rfind() grabbed wrong answer, find() fixed it.

9. **HPC speeds up testing.** 480 tests in 5.8 min vs ~280 min on CPU.

10. **Build backend classes first.** Retriever, Inference classes exist and work.

11. **Flask + Jinja is simpler than React.** Single Python file, no npm/node needed.

12. **Git LFS needs explicit pull in Docker.** Railway doesn't auto-pull LFS files.

13. **Tokenizers version must match training.** Caused cryptic deserialization errors.

14. **Gunicorn timeout must exceed response time.** 30s default too short for ML inference.

15. **Abstract base class pattern works.** Clean separation between Groq and custom model.

16. **groq package version matters.** 0.4.2 incompatible, 0.11.0+ works.

17. **Use --env-file for Docker.** Cleaner than passing -e for each variable.

---

*Document version: 17.0*
*Last updated: December 27, 2025*
