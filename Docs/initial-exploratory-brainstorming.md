# Initial Exploratory Brainstorming: Next LLM Project Stack

**Date:** December 20, 2025 (Updated December 26, 2025)
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

| Mode | LLM | Speed | Purpose |
|------|-----|-------|---------|
| Production | Groq API | ~1-2s | Fast UX |
| Demo | Custom 125.8M | ~35-37s | Prove ML skills to recruiters |

### Stack

| Component | Provider | Cost | Notes |
|-----------|----------|------|-------|
| **Frontend + Backend** | Railway (Hobby) | $5/month | Monorepo, always on |
| **LLM (Production)** | Groq API | $0 | Free tier sufficient |
| **LLM (Demo)** | Custom 125.8M | $0 | Lazy loaded, ~35-37s |
| **Embeddings** | Voyage AI | $0 | 200M free tokens |
| **Database** | Neon (free) | $0 | PostgreSQL + pgvector |

**Total: ~$5/month** (within $12 budget)

---

## Pipeline

```
User Question → Voyage AI embed → Neon vector search → Build prompt → LLM generates answer
                                                                         ↓
                                                              Groq (default) or Custom (demo)
```

---

## Custom Model (Demo Mode)

### Why Keep It?

Even though Groq handles production, the custom model demonstrates:
- Transformer architecture design
- Two-phase training pipeline
- HPC cluster experience (H100 GPUs)
- Data processing and cleaning

Recruiters can toggle "Demo Mode" to see the trained model in action.

### Configuration

| Parameter | Value |
|-----------|-------|
| Temperature | 0.2 |
| Top-k | 30 |
| Pass Rate | 100% |
| Keyword Score | 76-80% |
| Inference time | ~35-37s |

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

Model loads on first demo request, unloads after 5 min idle.

Cold start adds ~5s (negligible vs 35s inference).

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

### Generation: Groq API (Production)

| Setting | Value |
|---------|-------|
| Model | llama-3.3-70b-versatile |
| Free tier | 30 req/min, 14,400 req/day |
| Speed | ~1-2s per response |

### Generation: Custom Model (Demo)

| Setting | Value |
|---------|-------|
| Parameters | 125.8M |
| Speed | ~35-37s per response |
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
| Voyage AI | $0 (free tier) |
| Neon | $0 (free tier) |
| **Total** | **~$5/month** |

**Budget:** $12/month
**Actual:** ~$5/month
**Headroom:** ~$7/month

---

## Lessons Learned

1. **Don't trust synthetic data blindly.** ChatGPT generated 94% garbage.

2. **Two-phase training works.** Books for prose, context Q&A for RAG usage.

3. **125M params is the sweet spot.** 1.2M = gibberish, 125M = coherent.

4. **Lower temperature = better.** temp=0.2, top_k=30 achieved 100% pass rate.

5. **Lazy loading saves cost.** $2-3/month vs $6-8/month always loaded.

6. **IVFFlat index can miss results.** Exact search more reliable.

7. **Groq solves the speed problem.** Fast production UX, custom model for demo.

8. **Extraction function matters.** rfind() grabbed wrong answer, find() fixed it.

9. **HPC speeds up testing.** 480 tests in 5.8 min vs ~280 min on CPU.

---

*Document version: 12.0*
*Last updated: December 26, 2025*
