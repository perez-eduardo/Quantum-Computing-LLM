# Model Investigation Report

**Date:** December 21, 2025
**Last Updated:** December 31, 2025
**Purpose:** Document findings from model training, evaluation, and parameter tuning

---

## Current Status

**Training:** ✅ COMPLETE (v5, 140M params)
**RAG:** ✅ COMPLETE (100% retrieval accuracy)
**Parameter Tuning:** ✅ COMPLETE (temp=0.2, top_k=30)
**Extraction Fix:** ✅ COMPLETE (find() not rfind())
**Backend Classes:** ✅ COMPLETE (Retriever, BaseLLM, GroqInference, ModalInference)
**FastAPI App:** ✅ COMPLETE (dual-mode, suggested questions)
**Frontend:** ✅ COMPLETE (Flask + Jinja, model selector, health checks)
**Deployment:** ✅ COMPLETE (Railway, live)
**Groq Integration:** ✅ COMPLETE (deployed, ~2-3s response)
**Modal Deployment:** ✅ COMPLETE (T4 GPU, ~35-60s response)

---

## Live URLs

| Service | URL |
|---------|-----|
| Frontend | https://quantum-computing-llm.up.railway.app |
| Backend | https://quantum-computing-llm-backend.up.railway.app |
| API Docs | https://quantum-computing-llm-backend.up.railway.app/docs |
| Modal Query | https://perez-eduardo--quantum-llm-query.modal.run |
| Modal Health | https://perez-eduardo--quantum-llm-health.modal.run |

---

## Architecture

### Two LLM Modes

| Mode | LLM | Host | Speed | Status |
|------|-----|------|-------|--------|
| Production | Groq API (Llama 3.3 70B) | Groq Cloud | ~2-3s | ✅ Complete |
| Demo | Custom 140M | Modal (T4 GPU) | ~35-60s | ✅ Complete |

---

## Existing Backend Classes

### Retriever (`backend/scripts/retrieval.py`)

```python
class Retriever:
    def embed_query(query: str) -> List[float]
    def search(query: str, top_k: int = 5) -> List[Dict]
    def get_stats() -> Dict
```

### BaseLLM (`backend/scripts/base_inference.py`)

```python
class BaseLLM(ABC):
    @property
    def name(self) -> str
    def generate(context: str, question: str) -> str
    def extract_answer(generated_text: str) -> str
```

### GroqInference (`backend/scripts/groq_inference.py`)

```python
class GroqInference(BaseLLM):
    def __init__()
    def generate(context, question) -> str
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

## Key File Locations

| Component | Path |
|-----------|------|
| Model weights | `training/model/final_model.pt` |
| Tokenizer | `training/tokenizer/tokenizer.json` |
| Model architecture | `training/scripts/model.py` |
| Retriever class | `backend/scripts/retrieval.py` |
| Base LLM class | `backend/scripts/base_inference.py` |
| Groq inference | `backend/scripts/groq_inference.py` |
| Modal inference | `backend/scripts/modal_inference.py` |
| FastAPI config | `backend/app/config.py` |
| FastAPI main | `backend/app/main.py` |
| Flask app | `frontend/app.py` |
| Jinja template | `frontend/templates/index.html` |
| CSS styles | `frontend/static/css/style.css` |
| JavaScript | `frontend/static/js/main.js` |
| Modal deployment | `modal/inference.py` |

---

## Parameter Tuning Results (December 26, 2025)

### HPC Battery Test

Tested 24 parameter combinations (4 temps × 6 top_k) across 20 questions = 480 tests.

**Temps tested:** 0.1, 0.2, 0.3, 0.4
**Top_k tested:** 10, 20, 30, 40, 50, 60

**Top Results:**

| Parameters | Pass Rate | Keyword Score |
|------------|-----------|---------------|
| **temp=0.2, top_k=30** | **100%** | **80.5%** |
| temp=0.4, top_k=20 | 100% | 78.8% |
| temp=0.1, top_k=40 | 100% | 78.5% |
| temp=0.2, top_k=50 | 100% | 78.5% |
| temp=0.3, top_k=30 | 100% | 78.5% |
| temp=0.3, top_k=50 (old baseline) | 100% | 74.2% |

**Test time:** 5.8 minutes on H100 (vs ~280 min on CPU)

### Live Deployment Verification

Full RAG pipeline test (Voyage API + Neon DB + Custom Model):

| Parameters | Pass Rate | Keyword Score |
|------------|-----------|---------------|
| **temp=0.2, top_k=30** | **100%** | 76.2% |

**Winner:** temp=0.2, top_k=30

### Key Patterns

- Lower temperature (0.1-0.2) = more consistent, focused answers
- Moderate top_k (30-40) = best balance
- temp=0.4+ starts failing on some questions

---

## Groq Integration Results ✅

### Configuration

| Setting | Value |
|---------|-------|
| Model | llama-3.3-70b-versatile |
| Temperature | 0.2 |
| Max tokens | 300 |

### Test Results

20 quantum computing questions tested:

| Metric | Value |
|--------|-------|
| Pass rate | **100%** (20/20) |
| Avg response time | **~2-3s** |
| Answer quality | Beginner-friendly, accurate |

---

## Modal Deployment Results ✅

### Configuration

| Setting | Value |
|---------|-------|
| GPU | T4 (16GB VRAM) |
| Python | 3.10 |
| Tokenizers | 0.22.1 (must match training) |

### Performance

| Metric | Value |
|--------|-------|
| Cold start | 3-4 seconds |
| Warm inference | 5-10 seconds |
| Total response time | ~35-60 seconds |

---

## Extraction Function Fix (December 26, 2025)

### Problem

Model generates correct answer first, then continues generating more Q&A pairs.

Old extraction used `rfind("Answer:")` which grabbed the **LAST** (wrong) answer.

**Example:**
```
Question: What is a qubit? Answer: A qubit is the basic unit... Question: What is 2D? Answer: 2D qubits are...
                                   ^-- correct (first)                              ^-- wrong (last, grabbed by rfind)
```

### Fix

Changed `rfind()` to `find()` to get FIRST answer after the user's question.

**Result:** Previously failing questions now pass.

---

## Custom Model Architecture

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

**Important:** Model config is stored INSIDE the .pt file. Use `QuantumLLM.load(path, device)` to load both weights and config.

---

## Two-Phase Training

**Phase 1: Book Pretraining**
| Metric | Value |
|--------|-------|
| Data | combined_books_cleaned.txt (620K words) |
| Epochs | 17 |
| Final perplexity | **2.20** |
| Time | ~13 min on H100 |

**Phase 2: Context Q&A Fine-tuning**
| Metric | Value |
|--------|-------|
| Data | 28,071 context-format Q&A pairs |
| Epochs | 10 |
| Time | ~116 min on H100 |

---

## Training Results Comparison

| Metric | v1 | v3 | v4 | v5 |
|--------|----|----|----|----|
| Parameters | 1.2M | 1.2M | 1.2M | **140M** |
| Training format | Plain Q&A | Plain Q&A | Plain Q&A | **Context-aware** |
| Training data | 96K (garbage) | 24K | 26,764 | 28,071 |
| Final Perplexity | 15.55 | 89.63 | 91.80 | **2.20** |
| Pass Rate | 14.8% | 16.4% | 11.4% | **100%** |
| Boilerplate | 83.4% | 0% | 0% | 0% |

---

## RAG System ✅ COMPLETE

### Index Fix

IVFFlat approximate index was missing exact matches. Removed for exact search.

| Metric | Before | After |
|--------|--------|-------|
| Retrieval accuracy | 90% | **100%** |
| Search time | ~50ms | ~300ms |

### Database Contents

| Source | Count |
|--------|-------|
| claude | 14,400 |
| stackexchange | 10,673 |
| cot | 2,998 |
| **Total** | **28,071** |

---

## Inference Configuration

### Custom Model (Modal)

| Setting | Value |
|---------|-------|
| Temperature | 0.2 |
| Top-k | 30 |
| Speed | ~35-60s (Modal T4 GPU) |
| Host | Modal |

### Groq API ✅

| Setting | Value |
|---------|-------|
| Model | llama-3.3-70b-versatile |
| Temperature | 0.2 |
| Max tokens | 300 |
| Speed | ~2-3s |
| Host | Groq Cloud |

---

## API Endpoints (FastAPI)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Status, groq_available, modal_available |
| `/query` | POST | Question → answer, sources, response_time_ms, suggested_question, model_used |

### Request Format

```json
{
  "question": "What is a qubit?",
  "model": "groq"
}
```

### Response Format

```json
{
  "answer": "...",
  "sources": [...],
  "response_time_ms": 2500,
  "suggested_question": "...",
  "model_used": "groq"
}
```

### Suggested Question Feature

Extracts key terms from answer, scores retrieved questions by term matches, filters out questions >60% similar to original. No extra LLM call, no added latency.

---

## Frontend Features ✅

### Model Selector
- Dropdown in header (Groq vs Custom)
- Toast notification when Custom selected

### Health Check System
- Initial startup check
- Pre-query health check on every request
- Wake-up indicator for cold starts
- Specific error after 10 failed attempts

### Loading Indicators
- Groq: "Thinking..."
- Custom: Pipeline animation with 7 stages

### Error Handling
- Retry button on failed queries
- Timeout handling (30s Groq, 180s Custom)

### Modals
- Models modal: Groq vs Custom explanation
- About modal: Project details, tech stack, contact

---

## Lessons Learned

1. **Two-phase training works.** Book pretraining + context fine-tuning.

2. **140M params produces coherent text.** 1.2M was too small.

3. **Lower temperature = better.** temp=0.2 outperforms 0.3, 0.4.

4. **Moderate top_k = balanced.** top_k=30 beats both 20 and 50.

5. **Extraction function matters.** rfind() grabbed wrong answer, find() fixed it.

6. **Modal for GPU inference.** ~35-60s vs ~50-80s on Railway CPU.

7. **IVFFlat index caused retrieval failures.** Exact search fixed it.

8. **ChatGPT synthetic data was garbage.** 94% unusable.

9. **HPC accelerates testing.** 480 tests in 5.8 min vs ~280 min on CPU.

10. **Backend classes exist.** Retriever, BaseLLM, GroqInference, ModalInference ready.

11. **Flask + Jinja is simpler than React.** Single Python file, no npm/node.

12. **Git LFS needs explicit pull.** Railway doesn't auto-pull during build.

13. **Tokenizers version must match.** Training and deployment use same version.

14. **Gunicorn timeout must exceed response time.** Set 300s for safety.

15. **Groq integration works.** ~2-3s response time, clean architecture.

16. **Abstract base class pattern.** Clean separation between LLM implementations.

17. **groq package version matters.** 0.4.2 had httpx issues, 0.11.0+ works.

18. **Health check before every query.** Prevents 502 from sleeping servers.

19. **Model config inside .pt file.** QuantumLLM.load() extracts config automatically.

20. **Dockerfile timeout must match Procfile.** Both need 300s.

---

*Document version: 20.0*
