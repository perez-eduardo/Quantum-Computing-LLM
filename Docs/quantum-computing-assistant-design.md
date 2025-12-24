# Quantum Computing Assistant - Design Document

## Overview

Design specification for a web-based application that answers questions about foundational quantum computing concepts, powered by a custom-trained transformer model combined with RAG retrieval.

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
- Build complete training pipeline
- Train on real HPC hardware
- Deploy end-to-end ML system

The app itself is secondary. Recruiters evaluate the code, architecture decisions, and documentation.

**Target Audience:** Recruiters evaluating ML skills, students curious about quantum computing

---

## Current Status

| Component | Status |
|-----------|--------|
| Custom Model | ✅ Trained (v4) |
| RAG System | ✅ Complete (94% retrieval) |
| Backend | ⬜ In Progress |
| Frontend | ⬜ Pending |
| Deployment | ⬜ Pending |

**Next Action:** Connect RAG retrieval to model for end-to-end testing

---

## Topic Scope

**Focus:** Foundational quantum computing concepts for beginners (no math)

| Include | Exclude |
|---------|---------|
| What is a qubit | Dirac notation |
| Superposition (intuitive) | Matrix math |
| Entanglement (intuitive) | Complex numbers |
| Basic gates (what they do, not how) | Gate matrices |
| Quantum vs classical bits | Algorithms (Grover's, Shor's, etc.) |
| Why QC matters (applications overview) | Hardware implementation details |
| Common misconceptions | Bloch sphere math |

**Tone:** Explain like the reader is curious but not technical

---

## Tech Stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Frontend | Single HTML page | Minimal, self-contained |
| Backend | Python + FastAPI | RAG pipeline, model inference |
| Database | Neon PostgreSQL + pgvector | Free tier, 26,764 Q&A embeddings |
| Custom LLM | Trained transformer (~1.2M params) | Decoder-only, v4 |
| Training Data | Claude Q&A + Stack Exchange + CoT | 26,764 pairs |
| Embeddings | Voyage AI | voyage-3.5-lite, 200M free tokens |
| Training Compute | Oregon State HPC | H100 GPUs, SLURM scheduler |
| Hosting | Railway (Hobby) | $5/month, always on |

---

## Custom Transformer Model

### Architecture

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Type | Decoder-only (GPT-style) | Standard for text generation |
| Parameters | ~1.2M | Chinchilla-optimal for dataset |
| Vocabulary size | 16,000 | Custom BPE trained on corpus |
| Context length | 512 tokens | Sufficient for Q&A format |
| Layers | 4 | Scaled for 1.2M params |
| Attention heads | 4 | 16 dim per head |
| Embedding dimension | 64 | Scaled for 1.2M params |

### Training Status: ✅ Complete

### Training Data (v4)

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
| **v4** | **26,764** | **91.80** | **11.4%** |

### Model Files (Local)

| File | Location |
|------|----------|
| `final_model.pt` | `training/model/` |
| `best_model.pt` | `training/model/` |
| `config.json` | `training/model/` |
| `tokenizer.json` | `training/model/` |

### Tokenizer

**Approach:** Custom BPE tokenizer (16K vocab) trained on clean corpus.

| Token | ID | Purpose |
|-------|----|---------|
| `<pad>` | 0 | Padding for batching |
| `<eos>` | 1 | End of sequence |
| `<unk>` | 2 | Unknown token fallback |

---

## RAG System

### Status: ✅ Complete (94% retrieval)

### Purpose

The custom 1.2M parameter model has limited knowledge capacity. RAG compensates by retrieving relevant Q&A pairs at query time.

**Key insight from training:** Small models learn vocabulary, not reasoning. The model outputs quantum terminology but answers are incoherent. RAG provides the actual knowledge at inference time.

### Database Contents

| Source | Count |
|--------|-------|
| claude_synthetic | 15,000 |
| stackexchange | 9,008 |
| cot_reasoning | 2,756 |
| **Total** | **26,764** |

**Note:** Book chunks were removed. Q&A pairs provide better retrieval because they contain actual definitions rather than mentions.

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
| Test questions | 100 |
| Pass rate | 94% |
| Failures | 6 (data gaps, semantic edge cases) |

### Search Strategy

| Method | Details |
|--------|---------|
| Primary | Semantic search (pgvector cosine similarity) |
| Top-k | 3-5 chunks |
| Hybrid | Skipped (storage limit) |

---

## Inference Pipeline

```
1. User sends question
              │
              ▼
2. Generate query embedding (Voyage AI, input_type="query")
              │
              ▼
3. Retrieve top-k relevant Q&A pairs (pgvector)
              │
              ▼
4. Build prompt with context
              │
              ▼
5. Run inference: Custom transformer (or Groq fallback)
              │
              ▼
6. Return response to user
```

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
  "question": "What is quantum entanglement?"
}
```

Response:
```json
{
  "answer": "Quantum entanglement is a phenomenon where...",
  "sources": [
    {
      "source": "claude_synthetic",
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
| Response area | Model's answer with sources |
| Footer | Disclaimer, portfolio link |

---

## Hosting & Deployment

### Railway Configuration

| Setting | Value |
|---------|-------|
| Plan | Hobby ($5/month) |
| Structure | Monorepo (backend serves frontend) |
| Always on | Yes |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| VOYAGE_API_KEY | Embeddings |
| DATABASE_URL | Neon PostgreSQL connection |
| GROQ_API_KEY | Optional fallback LLM |

---

## Cost Summary

| Component | Monthly Cost |
|-----------|--------------|
| Railway (hosting) | $5 |
| Voyage AI (embeddings) | $0 (free tier) |
| Neon (database) | $0 (free tier) |
| HPC (training) | $0 (one-time) |
| **Total** | **$5/month** |

---

## Cost Protection

| Provider | Hard Cap | Action |
|----------|----------|--------|
| Railway | $10 | Settings > Usage > Hard limit |
| Voyage AI | $5 prepaid | Auto-recharge OFF |
| Neon | N/A | Auto-stops at 100 compute hours |

**Maximum exposure: $15/month**

---

## Out of Scope (MVP)

| Feature | Reason |
|---------|--------|
| Caching | Low traffic, adds complexity |
| Comparison mode | Not needed for portfolio |
| User level selection | 1.2M model too small |
| Hybrid search | Storage limit exceeded |

---

## Open Questions for End-to-End Testing

1. **Inference approach:** Custom model only, Groq only, or both?
2. **Test interface:** CLI script or API endpoint first?
3. **Prompt format:** How to structure retrieved context + question?

---

*Document version: 9.0*
*Last updated: December 24, 2025*
