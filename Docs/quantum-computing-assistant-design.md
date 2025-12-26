# Quantum Computing Assistant - Design Document

## Overview

Design specification for a web-based application that answers questions about foundational quantum computing concepts. Powered by Groq API for production, with custom-trained 125.8M parameter transformer available as demo mode.

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
| Custom Model (v5) | ✅ COMPLETE (125.8M params, 60% accuracy) |
| RAG System | ✅ COMPLETE (100% retrieval) |
| Parameter Tuning | ✅ COMPLETE (temp=0.3, top_k=20) |
| Backend | ⬜ IN PROGRESS |
| Frontend | ⬜ Pending |
| Deployment | ⬜ Pending |

**Next Action:** Integrate Groq API + Demo Mode

---

## Architecture

### Two LLM Modes

| Mode | LLM | Speed | Purpose |
|------|-----|-------|---------|
| **Production** | Groq API | ~1-2s | Fast UX (default) |
| **Demo** | Custom 125.8M | ~40-45s | Prove ML skills |

### Pipeline

```
User Question → Voyage AI embed → Neon vector search → Build prompt → LLM generates answer
                                                                         ↓
                                                              Groq (default) or Custom (demo)
```

### Stack

| Component | Provider | Speed | Cost |
|-----------|----------|-------|------|
| **Generation (Production)** | Groq API | ~1-2s | $0 (free tier) |
| **Generation (Demo)** | Custom 125.8M | ~40-45s | $0 (lazy loaded) |
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

## Custom Model (Demo Mode)

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

### Generation Config

| Parameter | Value |
|-----------|-------|
| Temperature | 0.3 |
| Top-k | 20 |
| Accuracy | 60% |

### Lazy Loading

| Setting | Value |
|---------|-------|
| Load trigger | First demo request |
| Unload trigger | 5 min idle |
| Cold start | ~5s |
| Inference | ~40-45s |
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
  "question": "What is quantum entanglement?",
  "demo_mode": false
}
```

Response:
```json
{
  "answer": "Quantum entanglement is a phenomenon where...",
  "mode": "groq",
  "response_time_ms": 1500,
  "sources": [
    {
      "source": "claude",
      "relevance": 0.89
    }
  ]
}
```

---

## UI Design

Single HTML page with minimal design.

| Element | Description |
|---------|-------------|
| Header | Title, brief description |
| Input area | Text input for questions |
| Response area | Answer with sources |
| **Demo toggle** | Switch between Groq and custom model |
| Footer | Disclaimer, portfolio link |

### Demo Mode UI

When demo mode is enabled:
- Show warning: "Demo mode uses custom model (~45s response time)"
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
| GROQ_API_KEY | Groq generation |
| VOYAGE_API_KEY | Embeddings |
| DATABASE_URL | Neon PostgreSQL connection |
| MODEL_PATH | Path to custom model weights |

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
3. ~~Tune generation params~~ ✅ Done (temp=0.3, top_k=20)
4. **Integrate Groq API** ⬜ Pending
5. **Implement lazy loading** ⬜ Pending
6. **Create FastAPI endpoints** ⬜ Pending
7. **Build frontend with demo toggle** ⬜ Pending
8. **Deploy to Railway** ⬜ Pending

---

*Document version: 14.0*
*Last updated: December 25, 2025*
