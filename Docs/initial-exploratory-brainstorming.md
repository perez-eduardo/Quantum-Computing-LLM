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
2. **RAG System** to supplement the model's knowledge

The custom transformer shows recruiters you understand ML internals. RAG provides relevant context at inference time.

**CRITICAL UPDATE (December 24):** Model must be trained with context-aware format to actually USE the RAG context. Plain Q&A training format cannot leverage retrieved content.

---

## Final Stack

| Component | Provider | Cost | Notes |
|-----------|----------|------|-------|
| **Frontend + Backend** | Railway (Hobby) | $5/month | Monorepo, always on |
| **LLM (Custom)** | Your trained transformer | $0 | 50M-100M params, context-aware |
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

### Architecture (UPDATED)

| Parameter | Original | Updated |
|-----------|----------|---------|
| Type | Decoder-only (GPT-style) | Same |
| Size | ~1.2M parameters | **50M-100M parameters** |
| Layers | 4 | TBD |
| Attention heads | 4 | TBD |
| Embedding dimension | 64 | TBD |
| Vocabulary | 16,000 (custom BPE) | Same |
| Context length | 512 tokens | Same |

**Why larger model:** 1.2M params cannot generate coherent sentences. Produces quantum jargon but gibberish structure. 50M-100M needed for readable output.

### Training Format (CRITICAL UPDATE)

**Old format (broken):**
```
Q: What is superposition?
A: Superposition allows...
```

**New format (context-aware):**
```
Context:
Q: What is entanglement?
A: Entanglement correlates two qubits...

Q: What is a qubit?
A: A qubit is the basic unit...

Question: What is superposition?
Answer: Superposition allows a qubit to be in multiple states...
```

Model learns: given context + question, generate answer.

### Training Data

#### Context-Format Dataset (NEW)

| Source | Rows | Context Type |
|--------|------|--------------|
| cot_qa_context.csv | 2,998 | Chain-of-thought reasoning |
| stackexchange_qa_context.csv | 10,673 | Tags + question body |
| claude_qa_batch1-38_context.csv | 15,000 | Template-based |
| **Total** | **28,671** | |

### Training Results (Pre-Redesign)

| Version | Data | Perplexity | Eval Score | Issue |
|---------|------|------------|------------|-------|
| v1 | 96K (garbage) | 15.55 | 14.8% | ChatGPT garbage |
| v3 | 24K (clean) | 89.63 | 16.4% | Plain format |
| v4 | 26,764 | 91.80 | 11.4% | Plain format, cannot use context |

**v5 (pending):** Context-aware format, 50M-100M params

### Compute: Oregon State University HPC

**Access confirmed.** Connection details:

| Field | Value |
|-------|-------|
| Host | `submit-b.hpc.engr.oregonstate.edu` |
| Username | `pereze4` |
| Scheduler | SLURM |
| Home Storage | 25 GB |
| HPC Share | 1.5 TB quota |

### Training Checklist

- [x] Finalize model size (1.2M parameters) - OUTDATED
- [x] Gather training data (quantum computing corpus)
- [x] Clean Stack Exchange data (filtered >1024 tokens)
- [x] Train custom tokenizer
- [x] Abandon ChatGPT data (94% garbage)
- [x] Generate Claude Q&A (15,000 pairs)
- [x] Obtain CoT Reasoning Dataset (2,998 pairs)
- [x] Train model v4 (plain format)
- [x] Set up RAG system
- [x] Embed Q&A pairs (26,764)
- [x] Test retrieval quality (94%)
- [x] **Add context to all datasets**
- [ ] **Update model architecture (50M-100M)**
- [ ] **Generate context-format training data**
- [ ] **Retrain model v5 (context-aware)**
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
- Near storage limit

---

### Frontend + Backend: Railway Hobby (Monorepo)

**Cost:** $5/month (includes $5 usage credit)

50M-100M parameter model is ~100-200MB. Still fits Railway deployment.

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

---

## Rejected Alternatives

### ChatGPT API for Synthetic Q&A
- **Status:** ABANDONED (December 22, 2025)
- Generated 85,643 Q&A pairs
- 83% contained repetitive boilerplate
- 59% were templated (only numbers changed)
- **Replaced with:** Claude-generated Q&A via chat

### Groq as Primary LLM
- **Status:** REJECTED (December 24, 2025)
- User requirement: Custom model must work, not just fallback
- Groq would make custom model pointless

### Plain Q&A Training Format
- **Status:** ABANDONED (December 24, 2025)
- Model cannot use RAG context if not trained on context format
- Must retrain with context-aware format

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
9. **Training format must match inference format** - Plain Q&A model cannot use context
10. **Small models cannot generate coherent text** - 1.2M params = gibberish, need 50M+

---

*Document version: 7.0*
*Last updated: December 24, 2025*
