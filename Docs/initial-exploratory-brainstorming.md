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
| **LLM (Custom)** | Your trained transformer | $0 | ~1.2M params, one-time training cost |
| **Embeddings** | Voyage AI | $0 | 200M free tokens |
| **Database** | Neon (free) | $0 | PostgreSQL + pgvector |
| **Training Compute** | Oregon State HPC | $0 | H100 GPUs available |

**Total: ~$5/month** (after one-time training)

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

> **🔄 PHASE 1 REDO (December 24, 2025)**
> 
> - New data source identified: CoT Reasoning Dataset (3,000 pairs)
> - Retraining with expanded dataset required
> - Target: Model v4 with ~27K Q&A pairs (~5.9M tokens)

#### v4 Dataset (Planned)

| Source | Count | Status |
|--------|-------|--------|
| Claude Q&A | 15,000 pairs | ✅ Ready |
| Stack Exchange (filtered) | 9,019 pairs | ✅ Ready |
| **CoT Reasoning Dataset** | **3,000 pairs** | ✅ Ready |
| Books | 633,562 words | ✅ Ready |
| **Total** | **~27,019 Q&A** | ⬜ Combine |

#### CoT Reasoning Dataset

| Property | Value |
|----------|-------|
| Location | `data/raw/source/CoT_Reasoning_Quantum_Physics_And_Computing.json` |
| Entries | 3,000 Q&A pairs |
| Features | Chain-of-thought reasoning, metadata (topic, difficulty) |
| License | MIT |

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

### Available GPU Partitions

| Partition | GPUs per Node | GPU Type | RAM | Time Limit |
|-----------|---------------|----------|-----|------------|
| **dgxh** | 16 | H100-40GB | 2 TB | 2 days |
| **dgx2** | 14-16 | V100 | 1.5 TB | 7 days |
| **gpu** | 8 | General | 760 GB | 7 days |
| **ampere** | 2 | A-series | 252 GB | 2 days |

**Recommended:** Use `dgxh` partition (H100s) for fastest training.

### Training Checklist

- [x] Finalize model size (1.2M parameters)
- [x] Gather training data (quantum computing corpus)
- [x] Clean Stack Exchange data (filtered >1024 tokens)
- [x] Train custom tokenizer
- [x] Upsample books 3x
- [x] Abandon ChatGPT data (94% garbage)
- [x] Generate Claude Q&A (15,000 pairs)
- [x] Combine final dataset (v3)
- [x] Train v3 with 10 epochs
- [x] Evaluate model (16.4% keyword match)
- [x] Verify data quality (0% boilerplate)
- [x] Obtain CoT Reasoning Dataset (3,000 pairs)
- [ ] Process CoT dataset
- [ ] Combine v4 dataset
- [ ] Train model v4
- [ ] Evaluate model v4
- [ ] Integrate with RAG system

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

**Cold start behavior:**
- Suspends after 5 minutes of inactivity
- Cold start latency: ~500ms-1 second
- Auto-wakes on first connection

**Setup complete:**
- pgvector extension enabled
- Connection tested

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

## Rejected Alternatives

### ChatGPT API for Synthetic Q&A
- **Status:** ABANDONED (December 22, 2025)
- Generated 85,643 Q&A pairs
- 83% contained repetitive boilerplate phrases
- 59% were templated (only numbers changed)
- After cleaning: only 4,808 usable (6%)
- **Replaced with:** Claude-generated Q&A via chat

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
7. **Look for existing quality datasets** - CoT Reasoning Dataset provides structured Q&A with reasoning

---

*Document version: 5.0*
*Last updated: December 24, 2025*
