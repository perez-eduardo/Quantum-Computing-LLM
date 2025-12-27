# Model Investigation Report

**Date:** December 21, 2025
**Updated:** December 27, 2025
**Purpose:** Document findings from model training, evaluation, and parameter tuning

---

## Current Status

**Training:** ✅ COMPLETE (v5, 125.8M params)
**RAG:** ✅ COMPLETE (100% retrieval accuracy)
**Parameter Tuning:** ✅ COMPLETE (temp=0.2, top_k=30)
**Extraction Fix:** ✅ COMPLETE (find() not rfind())
**Backend Classes:** ✅ COMPLETE (Retriever, BaseLLM, QuantumInference, GroqInference)
**FastAPI App:** ✅ COMPLETE (lazy loading, suggested questions, LLM selection)
**Frontend:** ✅ COMPLETE (Flask + Jinja)
**Deployment:** ✅ COMPLETE (Railway, live)
**Groq Integration:** ✅ COMPLETE (tested locally, 20/20 queries passed)

---

## Live URLs

| Service | URL |
|---------|-----|
| Frontend | https://quantum-computing-llm.up.railway.app |
| Backend | https://quantum-computing-llm-backend.up.railway.app |
| API Docs | https://quantum-computing-llm-backend.up.railway.app/docs |

---

## Architecture

### Two LLM Modes

| Mode | LLM | Speed | Status |
|------|-----|-------|--------|
| Production | Groq API (Llama 3.3 70B) | ~725ms | ✅ Complete |
| Demo | Custom 125.8M | ~50-80s | ✅ Deployed |

Custom model uses lazy loading to save cost (~$2-3/month vs $6-8/month).
Groq client is always loaded (lightweight, no lazy loading needed).

---

## Existing Backend Classes

### Retriever (`backend/scripts/retrieval.py`)

```python
class Retriever:
    def embed_query(query) -> List[float]   # Voyage AI, input_type="query"
    def search(query, top_k) -> List[Dict]  # Returns question, answer, source, similarity
    def get_stats() -> Dict                  # Database statistics
```

### BaseLLM (`backend/scripts/base_inference.py`)

```python
class BaseLLM(ABC):
    @property
    def name(self) -> str                    # "groq" or "custom"
    def generate(context, question) -> str
    def extract_answer(generated_text) -> str
```

### QuantumInference (`backend/scripts/inference.py`)

```python
class QuantumInference(BaseLLM):
    def __init__(model_path, tokenizer_path, device)
    def generate(context, question) -> str   # Builds flat prompt internally
    def extract_answer(generated_text) -> str  # Gets first answer after "Answer:"
    @property
    def name(self) -> str                    # Returns "custom"
```

### GroqInference (`backend/scripts/groq_inference.py`)

```python
class GroqInference(BaseLLM):
    def __init__()                           # Initializes Groq client
    def generate(context, question) -> str   # Chat completion API
    def extract_answer(generated_text) -> str  # Returns as-is (already clean)
    @property
    def name(self) -> str                    # Returns "groq"
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
| Custom inference | `backend/scripts/inference.py` |
| Groq inference | `backend/scripts/groq_inference.py` |
| FastAPI config | `backend/app/config.py` |
| FastAPI main | `backend/app/main.py` |
| Flask app | `frontend/app.py` |
| Jinja template | `frontend/templates/index.html` |
| CSS styles | `frontend/static/style.css` |

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

| Parameters | Pass Rate | Keyword Score | Retrieve | Generate | Total |
|------------|-----------|---------------|----------|----------|-------|
| **temp=0.2, top_k=30** | **100%** | 76.2% | 1.57s | 35.1s | 36.6s |
| temp=0.4, top_k=20 | 95% | 76.5% | 0.56s | 38.5s | 39.0s |

**Winner:** temp=0.2, top_k=30

### Key Patterns

- Lower temperature (0.1-0.2) = more consistent, focused answers
- Moderate top_k (30-40) = best balance
- temp=0.4+ starts failing on some questions

---

## Groq Integration Results (December 27, 2025)

### Configuration

| Setting | Value |
|---------|-------|
| Model | llama-3.3-70b-versatile |
| Temperature | 0.2 |
| Max tokens | 300 |

### Test Results

20 quantum computing questions tested via Docker:

| Metric | Value |
|--------|-------|
| Pass rate | **100%** (20/20) |
| Avg response time | **~725ms** |
| Answer quality | Beginner-friendly, accurate |

### Sample Response

```json
{
  "answer": "A qubit, short for quantum bit, is the basic unit of information in quantum computing. It's special because, unlike a classical bit that can only be 0 or 1, a qubit can be 0, 1, or both at the same time...",
  "sources": [
    {"question": "is a qubit just a tiny particle?", "source": "claude", "similarity": 0.7602},
    {"question": "What is a qubit?", "source": "claude", "similarity": 0.7377},
    {"question": "What is a quantum bit fundamentally?", "source": "claude", "similarity": 0.7346}
  ],
  "response_time_ms": 725,
  "model_loaded_fresh": false,
  "suggested_question": "can you explain qubits in simple terms?",
  "llm_used": "groq"
}
```

### Questions Tested

All passed:
- What is a qubit?
- What is superposition?
- What is quantum entanglement?
- What is a quantum gate?
- What is the Hadamard gate?
- What is quantum decoherence?
- What is a quantum circuit?
- What is Shor's algorithm?
- What is Grover's algorithm?
- What is quantum error correction?
- What is the Bloch sphere?
- What is quantum teleportation?
- What is a CNOT gate?
- What is quantum supremacy?
- What is a quantum register?
- What is quantum interference?
- What is quantum parallelism?
- What is a universal quantum gate set?
- What is measurement in quantum computing?
- How do quantum computers differ from classical computers?
- What are the applications of quantum computing?

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
| Parameters | **125,848,320 (125.8M)** |
| Layers | 12 |
| Attention heads | 12 |
| Embedding dimension | 768 |
| Feed-forward dimension | 3072 |
| Vocabulary | 16,384 (custom BPE) |
| Context length | 1024 tokens |

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
| Parameters | 1.2M | 1.2M | 1.2M | **125.8M** |
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

### Custom Model

| Setting | Value |
|---------|-------|
| Temperature | 0.2 |
| Top-k | 30 |
| Speed | ~50-80s (Railway CPU) |
| Loading | Lazy (5 min timeout) |

### Groq API ✅

| Setting | Value |
|---------|-------|
| Model | llama-3.3-70b-versatile |
| Temperature | 0.2 |
| Max tokens | 300 |
| Speed | ~725ms |
| Loading | Always loaded (lightweight) |

---

## API Endpoints (FastAPI)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Status, model_loaded, idle_seconds, groq_available |
| `/query` | POST | Question → answer, sources, response_time_ms, suggested_question, llm_used |

### Request Format

```json
{
  "question": "What is a qubit?",
  "use_groq": true
}
```

### Response Format

```json
{
  "answer": "...",
  "sources": [...],
  "response_time_ms": 725,
  "model_loaded_fresh": false,
  "suggested_question": "...",
  "llm_used": "groq"
}
```

### Suggested Question Feature

Extracts key terms from answer, scores retrieved questions by term matches, filters out questions >60% similar to original. No extra LLM call, no added latency.

---

## Frontend Features (Flask + Jinja)

### Welcome Screen
- Animated atom icon (CSS keyframes)
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

## Deployment Details (December 27, 2025)

### Railway Configuration

| Setting | Value |
|---------|-------|
| Plan | Hobby ($5/month) |
| Backend service | quantum-computing-llm-backend |
| Frontend service | quantum-computing-llm |

### Environment Variables

**Backend:**
- VOYAGE_API_KEY
- DATABASE_URL
- GROQ_API_KEY (pending Railway deployment)

**Frontend:**
- BACKEND_URL=https://quantum-computing-llm-backend.up.railway.app

### Deployment Issues Resolved

| Issue | Solution |
|-------|----------|
| Git LFS pointer file (130 bytes) | Clone repo + `git lfs pull` in Dockerfile |
| Tokenizers version mismatch | Updated 0.15.0 → 0.22.1 |
| Gunicorn timeout (30s default) | Set `--timeout 600` in Procfile |
| groq proxy error | Updated groq==0.4.2 → groq>=0.11.0 |

---

## Lessons Learned

1. **Two-phase training works.** Book pretraining + context fine-tuning.

2. **125M params produces coherent text.** 1.2M was too small.

3. **Lower temperature = better.** temp=0.2 outperforms 0.3, 0.4.

4. **Moderate top_k = balanced.** top_k=30 beats both 20 and 50.

5. **Extraction function matters.** rfind() grabbed wrong answer, find() fixed it.

6. **Custom model inference:** ~50-80s per question on Railway CPU.

7. **Lazy loading saves cost.** Load on demand, unload after idle.

8. **IVFFlat index caused retrieval failures.** Exact search fixed it.

9. **ChatGPT synthetic data was garbage.** 94% unusable.

10. **HPC accelerates testing.** 480 tests in 5.8 min vs ~280 min on CPU.

11. **Backend classes exist.** Retriever, BaseLLM, QuantumInference, GroqInference ready to use.

12. **Flask + Jinja is simpler than React.** Single Python file, no npm/node needed.

13. **Git LFS needs explicit pull.** Railway doesn't auto-pull LFS files during build.

14. **Tokenizers version must match.** Training and deployment must use same version.

15. **Gunicorn timeout must exceed response time.** Set 10x expected for safety.

16. **Groq integration works.** ~725ms response time, clean architecture.

17. **Abstract base class pattern.** Clean separation between LLM implementations.

18. **groq package version matters.** 0.4.2 had httpx proxy issues, 0.11.0+ works.

---

*Document version: 19.0*
*Last updated: December 27, 2025*
