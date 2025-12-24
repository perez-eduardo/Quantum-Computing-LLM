# Initial Exploratory Brainstorming: Next LLM Project Stack

**Date:** December 20, 2025 (Updated December 24, 2025)
**Project:** Quantum Computing LLM  
**Purpose:** Portfolio demonstration piece  
**Expected Traffic:** Minimal (recruiters, students)

---

## Problem Statement

The Philippine Legal Assistant project revealed infrastructure limitations:

1. **Groq free tier:** Shared limits across all users caused rate limiting
2. **Render free tier:** 512MB RAM cannot handle local Sentence Transformers model

Need a stack that is:
- Cost-optimized (minimal monthly spend)
- Reliable (no shared rate limits)
- Fast (minimal cold starts)

---

## Project Architecture

This project combines two components:

1. **Custom Transformer** (trained from scratch) for portfolio demonstration
2. **RAG System** to supplement the small model's limited knowledge

The custom transformer shows recruiters you understand ML internals. RAG compensates for the small model's limitations by retrieving relevant documents.

---

## Final Stack

| Component | Provider | Cost | Notes |
|-----------|----------|------|-------|
| **Frontend + Backend** | Railway (Hobby) | $5/month | Monorepo, always on |
| **LLM (Custom)** | Your trained transformer | $0 | ~1.2M params, trained on HPC |
| **Embeddings** | Voyage AI | $0 | 200M free tokens |
| **Database** | Neon (free) | $0 | PostgreSQL + pgvector |
| **Training Compute** | Oregon State HPC | $0 | H100 GPUs available |

**Total: ~$5/month**

---

## Custom Transformer Training

### Purpose

- Demonstrate understanding of transformer architecture
- Show ability to build a complete training pipeline
- Create a "fully yours" component for the portfolio
- Combined with RAG for actual usefulness

### Architecture

| Parameter | Value |
|-----------|-------|
| Type | Decoder-only (GPT-style) |
| Size | ~1.2M parameters |
| Layers | 4 |
| Attention heads | 4 |
| Embedding dimension | 64 |
| Vocabulary | 16,000 (custom BPE) |
| Context length | 512 tokens |

### Training Data

#### v4 Dataset (Final)

| Source | Count | Status |
|--------|-------|--------|
| Claude Q&A | 15,000 pairs | ✅ Complete |
| Stack Exchange (filtered) | 9,008 pairs | ✅ Complete |
| CoT Reasoning Dataset | 2,756 pairs | ✅ Complete |
| **Total** | **26,764 Q&A** | ✅ Complete |

### Training Results

| Version | Data | Perplexity | Eval Score |
|---------|------|------------|------------|
| v1 | 96K (garbage) | 15.55 | 14.8% |
| v3 | 24K (clean) | 89.63 | 16.4% |
| v4 | 26,764 (expanded) | 91.80 | 11.4% |

### Compute: Oregon State University HPC

**Access confirmed.** Connection details:

| Field | Value |
|-------|-------|
| Host | `submit-b.hpc.engr.oregonstate.edu` |
| Username | `pereze4` |
| Scheduler | SLURM |
| Home Storage | 25 GB |
| HPC Share | 1.5 TB quota |

**Connect from Windows PowerShell:**
```powershell
ssh pereze4@submit-b.hpc.engr.oregonstate.edu
```

### Training Checklist

- [x] Finalize model size (1.2M parameters)
- [x] Gather training data (quantum computing corpus)
- [x] Clean Stack Exchange data (filtered >1024 tokens)
- [x] Train custom tokenizer
- [x] Abandon ChatGPT data (94% garbage)
- [x] Generate Claude Q&A (15,000 pairs)
- [x] Obtain CoT Reasoning Dataset (2,756 pairs)
- [x] Combine v4 dataset (26,764 pairs)
- [x] Train model v4
- [x] Evaluate model v4
- [x] Download model files
- [x] Set up RAG system
- [x] Embed Q&A pairs (26,764)
- [x] Test retrieval quality (94%)
- [ ] Connect RAG to model
- [ ] End-to-end testing

---

## Component Details

### Embeddings: Voyage AI

**Why Voyage AI over alternatives:**

| Provider | Free Tokens | Free RPM | Paid Rate |
|----------|-------------|----------|-----------|
| Jina AI | 10M (one-time) | 10-30 RPM | $0.02/1M |
| Voyage AI | 200M | 3 RPM (free), 2000 RPM (with card) | $0.02/1M |
| Google AI | Gutted Dec 2025 | Unreliable | Unknown |

**Critical setup step:**
- Must add payment method to unlock 2000 RPM
- Without card: only 3 requests/minute (unusable)
- With card: 2000 RPM, still uses free tokens

**Cost protection:**
- Prepaid $5 credits with auto-recharge OFF
- Acts as hard spending cap

---

### Database: Neon Free Tier

**Limits:**
| Resource | Limit |
|----------|-------|
| Storage | 0.5 GB per project |
| Compute | 100 CU-hours/month |
| Egress | 5 GB/month |

**Current usage:**
- 26,764 Q&A embeddings stored
- Near storage limit (required removing book chunks)

**Cold start behavior:**
- Suspends after 5 minutes of inactivity
- Cold start latency: ~500ms-1 second
- Auto-wakes on first connection

---

### Frontend + Backend: Railway Hobby (Monorepo)

**Cost:** $5/month (includes $5 usage credit)

**Architecture:**
```
/your-project
  /frontend    (Single HTML page)
  /backend     (FastAPI)
```

FastAPI serves the frontend as static files. Single service, single URL.

---

## RAG System Status

### Retrieval Quality: 94%

| Test | Result |
|------|--------|
| Questions tested | 100 |
| Pass rate | 94% |
| Failures | 6 (data gaps, semantic edge cases) |

### Database Contents

| Content | Count |
|---------|-------|
| Q&A embeddings | 26,764 |
| Book chunks | 0 (removed) |

**Why book chunks removed:** Q&A pairs have actual definitions. Book chunks mentioned terms without defining them, causing retrieval failures.

---

## Rejected Alternatives

### ChatGPT API for Synthetic Q&A
- **Status:** ABANDONED (December 22, 2025)
- Generated 85,643 Q&A pairs
- 83% contained repetitive boilerplate phrases
- 59% were templated (only numbers changed)
- After cleaning: only 4,808 usable (6%)
- **Replaced with:** Claude-generated Q&A via chat

### Hybrid Search (BM25 + Semantic)
- **Status:** SKIPPED (December 24, 2025)
- Would improve retrieval from 94% to ~96-97%
- Requires tsvector column and GIN index
- Neon storage limit (512MB) exceeded
- 94% is sufficient for portfolio project

### Google AI Studio (LLM + Embeddings)
- **Status:** Gutted in December 2025
- Gemini 2.5 Pro removed from free tier
- Unreliable for production use

### Supabase (Database)
- Pauses after 7 days of inactivity
- Requires manual restart
- Bad for portfolio with sporadic traffic

---

## Cost Summary

| Component | Monthly Cost |
|-----------|--------------|
| Railway (Frontend + Backend) | $5 |
| Voyage AI (Embeddings) | $0 |
| Neon (Database) | $0 |
| HPC (Training) | $0 (one-time) |
| **Total** | **$5/month** |

---

## Lessons Learned

1. **Don't trust "free" shared limits** - Groq free tier failed due to global exhaustion
2. **Check RAM requirements** - Local models need more than free tier provides
3. **Set spending caps immediately** - Never deploy without hard limits
4. **Inspect data at every step** - Don't process garbage through your pipeline
5. **Don't trust synthetic data blindly** - ChatGPT generated 94% garbage
6. **Verify after training** - Run boilerplate and quality checks on outputs
7. **Q&A pairs beat book chunks for RAG** - Definitions > mentions
8. **94% retrieval is good enough** - Diminishing returns on further optimization
9. **Storage limits matter** - Neon 512MB constrains what you can store

---

*Document version: 6.0*
*Last updated: December 24, 2025*
