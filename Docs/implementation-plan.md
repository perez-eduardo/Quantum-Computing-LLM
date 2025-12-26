# Quantum Computing Assistant - Implementation Plan

**Last Updated:** December 26, 2025

**Related Documents:**
- Design Document: `quantum-computing-assistant-design.md`
- Infrastructure Planning: `initial-exploratory-brainstorming.md`
- Model Investigation Report: `model_investigation_report.md`

---

## Current Status

**Phase 1:** Training Pipeline - ✅ COMPLETE (v5 trained, 100% pass rate)
**Phase 2:** RAG System - ✅ COMPLETE (100% retrieval accuracy)
**Phase 3:** Backend - ⬜ IN PROGRESS

**Next Action:** Integrate Groq API + Demo Mode with lazy loading

---

## Architecture (December 26, 2025)

### Two LLM Modes

| Mode | LLM | Speed | Purpose |
|------|-----|-------|---------|
| Production | Groq API | ~1-2s | Fast UX |
| Demo | Custom 125.8M | ~35-37s | Prove ML skills |

### Pipeline

```
User Question → Voyage AI embed → Neon vector search → Build prompt → LLM generates answer
                                                                         ↓
                                                              Groq (default) or Custom (demo)
```

### Custom Model Config

| Parameter | Value |
|-----------|-------|
| Temperature | 0.2 |
| Top-k | 30 |
| Loading | Lazy (load on first demo request) |
| Timeout | Unload after 5 min idle |

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

### Custom Model Training

| Task | Status | Notes |
|------|--------|-------|
| Train 125.8M transformer | ✅ Done | Two-phase training |
| Tune generation params | ✅ Done | temp=0.2, top_k=30 |
| Achieve 100% pass rate | ✅ Done | Best config |
| Fix extraction function | ✅ Done | First answer, not last |

---

## What Is Next

### Phase 3: Backend

| Task | Priority | Status |
|------|----------|--------|
| Add GROQ_API_KEY to .env | High | ⬜ Pending |
| Create Groq generation module | High | ⬜ Pending |
| Create RAG retrieval module | High | ⬜ Pending |
| Implement lazy model loading | High | ⬜ Pending |
| Add demo mode toggle endpoint | High | ⬜ Pending |
| Create FastAPI endpoints | High | ⬜ Pending |
| Test end-to-end flow | High | ⬜ Pending |

### Phase 4: Frontend

| Task | Priority | Status |
|------|----------|--------|
| Single HTML page | Medium | ⬜ Pending |
| API integration | Medium | ⬜ Pending |
| Demo mode toggle UI | Medium | ⬜ Pending |

### Phase 5: Deployment

| Task | Priority | Status |
|------|----------|--------|
| Deploy to Railway | Medium | ⬜ Pending |
| Set spending caps | Medium | ⬜ Pending |

---

## Key Findings

1. **ChatGPT synthetic data was 94% garbage.** Abandoned entirely.

2. **Claude Q&A generation works.** 15,000 pairs verified.

3. **IVFFlat index caused retrieval failures.** Exact search fixed it.

4. **Custom model best params:** temp=0.2, top_k=30 (100% pass rate, 76-80% keyword score).

5. **Extraction function bug:** Was using rfind() (last answer), fixed to find() (first answer).

6. **Lazy loading saves cost.** ~$2-3/month vs $6-8/month always loaded.

7. **Custom model inference:** ~35-37s per question on CPU.

8. **HPC battery test:** 480 tests in 5.8 minutes on H100 (vs ~280 min on CPU).

---

*Document version: 17.0*
