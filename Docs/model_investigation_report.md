# Model Investigation Report

**Date:** December 21, 2025
**Updated:** December 26, 2025
**Purpose:** Document findings from model training, evaluation, and parameter tuning

---

## Current Status

**Training:** ✅ COMPLETE (v5, 125.8M params)
**RAG:** ✅ COMPLETE (100% retrieval accuracy)
**Parameter Tuning:** ✅ COMPLETE (temp=0.2, top_k=30)
**Extraction Fix:** ✅ COMPLETE (find() not rfind())
**Backend Classes:** ✅ COMPLETE (Retriever, QuantumInference, Pipeline)
**FastAPI App:** ⬜ IN PROGRESS

---

## Architecture (December 26, 2025)

### Two LLM Modes (Implementation Order)

| Mode | LLM | Speed | Status |
|------|-----|-------|--------|
| Custom | Custom 125.8M | ~35-37s | ⬜ Implement first |
| Production | Groq API | ~1-2s | ⬜ Add later |

Custom model uses lazy loading to save cost (~$2-3/month vs $6-8/month).

---

## Existing Backend Classes

### Retriever (`backend/scripts/retrieval.py`)

```python
class Retriever:
    def embed_query(query) -> List[float]   # Voyage AI, input_type="query"
    def search(query, top_k) -> List[Dict]  # Returns question, answer, source, similarity
    def get_stats() -> Dict                  # Database statistics
```

### QuantumInference (`backend/scripts/inference.py`)

```python
class QuantumInference:
    def __init__(model_path, tokenizer_path, device)
    def generate(prompt, max_new_tokens=150, temperature=0.2, top_k=30) -> str
    def extract_answer(generated_text) -> str  # Gets first answer after "Answer:"
```

### QuantumRAGPipeline (`backend/scripts/pipeline.py`)

```python
class QuantumRAGPipeline:
    def __init__(model_path, tokenizer_path, device)
    def query(question) -> Dict  # Returns answer, sources, suggested_questions
```

---

## Key File Locations

| Component | Path |
|-----------|------|
| Model weights | `training/model/final_model.pt` |
| Tokenizer | `training/tokenizer/tokenizer.json` |
| Model architecture | `training/scripts/model.py` |
| Retriever class | `backend/scripts/retrieval.py` |
| Inference class | `backend/scripts/inference.py` |
| Pipeline class | `backend/scripts/pipeline.py` |
| Parameter verification | `backend/scripts/verify_params.py` |

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
| Speed | ~35-37s |
| Loading | Lazy (5 min timeout) |

### Groq API (Later)

| Setting | Value |
|---------|-------|
| Model | llama-3.3-70b-versatile |
| Speed | ~1-2s |

---

## Lessons Learned

1. **Two-phase training works.** Book pretraining + context fine-tuning.

2. **125M params produces coherent text.** 1.2M was too small.

3. **Lower temperature = better.** temp=0.2 outperforms 0.3, 0.4.

4. **Moderate top_k = balanced.** top_k=30 beats both 20 and 50.

5. **Extraction function matters.** rfind() grabbed wrong answer, find() fixed it.

6. **Custom model inference:** ~35-37s per question (improved from 40-45s estimate).

7. **Lazy loading saves cost.** Load on demand, unload after idle.

8. **IVFFlat index caused retrieval failures.** Exact search fixed it.

9. **ChatGPT synthetic data was garbage.** 94% unusable.

10. **HPC accelerates testing.** 480 tests in 5.8 min vs ~280 min on CPU.

11. **Backend classes exist.** Retriever, QuantumInference, Pipeline ready to use.

---

*Document version: 16.0*
*Last updated: December 26, 2025*
