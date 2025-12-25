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

1. **Custom Transformer** (125.8M params, trained from scratch) for portfolio demonstration
2. **RAG System** to provide context at inference time

The custom transformer shows recruiters you understand ML internals. RAG provides relevant Q&A pairs as context.

**Model requires context to function.** Without RAG, outputs are gibberish. This is by design.

---

## Final Stack

| Component | Provider | Cost | Notes |
|-----------|----------|------|-------|
| **Frontend + Backend** | Railway (Hobby) | $5/month | Monorepo, always on |
| **LLM (Custom)** | Your trained transformer | $0 | 125.8M params, context-aware |
| **Embeddings** | Voyage AI | $0 | 200M free tokens |
| **Database** | Neon (free) | $0 | PostgreSQL + pgvector |
| **Training Compute** | Oregon State HPC | $0 | H100 GPUs |

**Total: ~$5/month**

---

## Custom Transformer Training ✅ COMPLETE

### Architecture (Final)

| Parameter | Value |
|-----------|-------|
| Type | Decoder-only (GPT-style) |
| Parameters | **125,848,320 (125.8M)** |
| Layers | 12 |
| Attention heads | 12 |
| Embedding dimension | 768 |
| Feed-forward dimension | 3072 |
| Vocabulary | 16,384 (custom BPE) |
| Context length | 1024 tokens |

### Two-Phase Training Approach

**Phase 1: Book Pretraining**
- Data: 620K words from 5 cleaned textbooks
- Purpose: Learn coherent prose generation
- Result: Perplexity 2.20 (17 epochs)

**Phase 2: Context Q&A Fine-tuning**
- Data: 28,071 context-format Q&A pairs
- Purpose: Learn to use RAG context
- Result: 64% keyword accuracy (10 epochs)

### Training Results

| Version | Params | Format | Perplexity | Accuracy | Status |
|---------|--------|--------|------------|----------|--------|
| v1 | 1.2M | Plain Q&A | 15.55 | 14.8% | ❌ Garbage data |
| v3 | 1.2M | Plain Q&A | 89.63 | 16.4% | ❌ Cannot use context |
| v4 | 1.2M | Plain Q&A | 91.80 | 11.4% | ❌ Cannot use context |
| **v5** | **125.8M** | **Context-aware** | **2.20** | **64%** | **✅ Production** |

### Training Data

| Source | Rows | Context Type |
|--------|------|--------------|
| claude_qa_context.csv | 14,400 | Topic-matched relevant Q&A pairs |
| cot_qa_context.csv | 2,998 | Chain-of-thought reasoning |
| stackexchange_qa_context.csv | 10,673 | Tags + question body |
| combined_books_cleaned.txt | 620K words | Phase 1 pretraining |
| **Total Q&A** | **28,071** | |

### Compute: Oregon State University HPC

| Field | Value |
|-------|-------|
| Host | `submit-b.hpc.engr.oregonstate.edu` |
| Username | `pereze4` |
| Scheduler | SLURM |
| GPU | H100 80GB |
| Phase 1 time | ~13 min |
| Phase 2 time | ~116 min |

### Training Checklist ✅

- [x] Gather training data (quantum computing corpus)
- [x] Clean Stack Exchange data
- [x] Clean book texts (620K words)
- [x] Train custom tokenizer (16K vocab)
- [x] Abandon ChatGPT data (94% garbage)
- [x] Generate Claude Q&A (15,000 pairs)
- [x] Obtain CoT Reasoning Dataset (2,998 pairs)
- [x] Add context to all datasets
- [x] Implement 125.8M param architecture
- [x] Phase 1: Book pretraining (17 epochs, perplexity 2.20)
- [x] Phase 2: Context Q&A fine-tuning (10 epochs)
- [x] Evaluate model (64% keyword accuracy)
- [x] Set up RAG system (94% retrieval)
- [ ] Connect RAG to model
- [ ] End-to-end testing
- [ ] Deploy to Railway

---

## Component Details

### Embeddings: Voyage AI

| Provider | Free Tokens | Free RPM | Paid Rate |
|----------|-------------|----------|-----------|
| Voyage AI | 200M | 2000 RPM (with card) | $0.02/1M |

**Critical:** Must add payment method to unlock 2000 RPM

### Database: Neon Free Tier

| Resource | Limit | Current Usage |
|----------|-------|---------------|
| Storage | 0.5 GB | 26,764 embeddings |
| Compute | 100 CU-hours/month | Normal |

### Frontend + Backend: Railway Hobby

| Setting | Value |
|---------|-------|
| Cost | $5/month |
| Model size | ~500MB (125.8M params) |
| Always on | Yes |

---

## RAG System Status ✅ COMPLETE

### Retrieval Quality: 94%

| Metric | Value |
|--------|-------|
| Q&A embeddings | 26,764 |
| Test pass rate | 94% |

---

## Rejected Alternatives

### ChatGPT API for Synthetic Q&A
- **Status:** ABANDONED
- 94% garbage rate (boilerplate, templates)
- **Replaced with:** Claude-generated Q&A

### Plain Q&A Training Format
- **Status:** ABANDONED
- Model cannot use RAG context if not trained on context format
- **Replaced with:** Two-phase context-aware training

### 1.2M Parameter Model
- **Status:** ABANDONED
- Cannot generate coherent sentences
- **Replaced with:** 125.8M parameters

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

1. **Don't trust synthetic data blindly.** ChatGPT generated 94% garbage.

2. **Two-phase training works.** Books for prose, context Q&A for RAG usage.

3. **125M params is the sweet spot.** 1.2M = gibberish, 125M = coherent with context.

4. **Context format must match inference.** Train on what you'll use.

5. **Topic-matched context beats random.** Relevant Q&A pairs simulate RAG.

6. **Model needs context to function.** Without RAG, outputs are gibberish. By design.

7. **64% keyword accuracy is achievable.** Major improvement from 11.4%.

8. **Book cleaning matters.** Remove copyright, TOC, spam.

9. **SLURM time limits matter.** Set --time=04:00:00 for safety.

10. **PYTHONUNBUFFERED=1 for real-time output.** Otherwise logs are delayed.

---

*Document version: 8.0*
*Last updated: December 24, 2025*
