# Quantum Computing Assistant - Design Document

## Overview

Design specification for a web-based application that answers questions about foundational quantum computing concepts, powered by a custom-trained 125.8M parameter transformer combined with RAG retrieval.

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
| Custom Model (v5) | ✅ COMPLETE (125.8M params, 64% accuracy) |
| Context-Format Data | ✅ COMPLETE (28,071 rows) |
| RAG System | ✅ COMPLETE (94% retrieval) |
| Backend | ⬜ IN PROGRESS |
| Frontend | ⬜ Pending |
| Deployment | ⬜ Pending |

**Next Action:** Connect RAG to model for end-to-end testing

---

## Model v5 Architecture ✅ COMPLETE

### Parameters

| Parameter | Value |
|-----------|-------|
| Type | Decoder-only transformer |
| Total parameters | **125,848,320 (125.8M)** |
| Layers | 12 |
| Attention heads | 12 |
| Embedding dimension | 768 |
| Feed-forward dimension | 3072 |
| Vocabulary size | 16,384 (custom BPE) |
| Context length | 1024 tokens |
| Dropout | 0.1 |

### Features
- RMSNorm (instead of LayerNorm)
- Rotary Position Embeddings (RoPE)
- SwiGLU activation
- Weight tying (embedding and output)

### Two-Phase Training

**Phase 1: Book Pretraining**
| Metric | Value |
|--------|-------|
| Data | 620K words from cleaned textbooks |
| Tokens | 970,811 |
| Epochs | 17 |
| Final perplexity | **2.20** |
| Time | ~13 min on H100 |

**Phase 2: Context Q&A Fine-tuning**
| Metric | Value |
|--------|-------|
| Data | 28,071 context-format Q&A pairs |
| Epochs | 10 |
| Time | ~116 min on H100 |

### Evaluation Results

| Test Type | Result |
|-----------|--------|
| With context | **64% keyword accuracy** |
| Without context | Gibberish (expected) |

**Model requires RAG context to function.** This is by design.

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

## Tech Stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Frontend | Single HTML page | Minimal, self-contained |
| Backend | Python + FastAPI | RAG pipeline, model inference |
| Database | Neon PostgreSQL + pgvector | Free tier, 26,764 Q&A embeddings |
| Custom LLM | Trained transformer | **125.8M params, context-aware** |
| Training Data | Context-format Q&A | 28,071 pairs |
| Embeddings | Voyage AI | voyage-3.5-lite, 200M free tokens |
| Training Compute | Oregon State HPC | H100 GPUs, SLURM scheduler |
| Hosting | Railway (Hobby) | $5/month, always on |

---

## Training Data

### Context-Format Q&A (Phase 2)

| Source | Rows | Context Type |
|--------|------|--------------|
| claude_qa_context.csv | 14,400 | Topic-matched relevant Q&A pairs |
| cot_qa_context.csv | 2,998 | Chain-of-thought reasoning |
| stackexchange_qa_context.csv | 10,673 | Tags + question body |
| **Total** | **28,071** | |

### Books (Phase 1)

| Source | Words |
|--------|-------|
| combined_books_cleaned.txt | 620,455 |

5 textbooks cleaned of copyright notices, TOC, spam.

---

## RAG System ✅ COMPLETE

### Purpose

The model requires context to generate coherent answers. RAG retrieves relevant Q&A pairs at query time.

### Database Contents

| Source | Count |
|--------|-------|
| claude_synthetic | 15,000 |
| stackexchange | 9,008 |
| cot_reasoning | 2,756 |
| **Total** | **26,764** |

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
| Pass rate | **94%** |

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
4. Build context-aware prompt:
   
   Context: Q: [retrieved pair 1] A: [answer 1]
   Q: [retrieved pair 2] A: [answer 2]
   
   Question: [user question]
   Answer:
              │
              ▼
5. Run inference: Custom transformer (125.8M params)
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

## Model Files (HPC)

| File | Location | Description |
|------|----------|-------------|
| `final_model.pt` | `model/` | **Production model** (~500MB) |
| `phase1_best.pt` | `model/` | Book pretraining checkpoint |
| `phase2_best.pt` | `model/` | Context fine-tuning checkpoint |
| `config.json` | `model/` | Model configuration |
| `tokenizer.json` | root | BPE tokenizer (16K vocab) |

---

## Hosting & Deployment

### Railway Configuration

| Setting | Value |
|---------|-------|
| Plan | Hobby ($5/month) |
| Structure | Monorepo (backend serves frontend) |
| Always on | Yes |
| Model size | ~500MB (125.8M params) |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| VOYAGE_API_KEY | Embeddings |
| DATABASE_URL | Neon PostgreSQL connection |

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
| User level selection | Single model |
| Hybrid search | Storage limit exceeded |
| Groq fallback | Custom model is the point |

---

## Next Steps

1. ~~Update model architecture~~ ✅ Done (125.8M params)
2. ~~Train Phase 1 (books)~~ ✅ Done (perplexity 2.20)
3. ~~Train Phase 2 (context Q&A)~~ ✅ Done (64% accuracy)
4. **Download model from HPC** ⬜ Pending
5. **Connect RAG to model** ⬜ Pending
6. **End-to-end testing** ⬜ Pending
7. **Deploy to Railway** ⬜ Pending

---

*Document version: 11.0*
*Last updated: December 24, 2025*
