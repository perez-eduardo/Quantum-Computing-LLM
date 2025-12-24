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

**Target Audience:** Recruiters evaluating ML skills, students curious about quantum computing

---

## Current Status

| Component | Status |
|-----------|--------|
| Custom Model (v4) | ⚠️ Cannot use context (needs retraining) |
| Context-Format Data | ✅ Complete (28,671 rows) |
| RAG System | ✅ Complete (94% retrieval) |
| Backend | ⬜ Blocked (awaiting model) |
| Frontend | ⬜ Pending |
| Deployment | ⬜ Pending |

**Next Action:** Retrain model with context-aware format (50M-100M params)

---

## CRITICAL: Design Flaw & Resolution

### Problem Discovered (December 24, 2025)

The v4 model was trained on plain Q&A format but RAG provides context at inference:

**Training format (v4):**
```
Q: What is superposition?
A: Superposition allows...
```

**Inference format (with RAG):**
```
Context: [retrieved Q&A pairs]
Question: What is superposition?
```

**Result:** Model ignores context entirely - never learned the format.

### Resolution

1. **Retrain with context-aware format:**
```
Context:
Q: What is entanglement?
A: Entanglement correlates two qubits...

Q: What is a qubit?
A: A qubit is the basic unit...

Question: What is superposition?
Answer: Superposition allows a qubit to be in multiple states...
```

2. **Scale to 50M-100M parameters** (1.2M cannot generate coherent text)

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
| Custom LLM | Trained transformer | **50M-100M params, context-aware** |
| Training Data | Context-format Q&A | 28,671 pairs |
| Embeddings | Voyage AI | voyage-3.5-lite, 200M free tokens |
| Training Compute | Oregon State HPC | H100 GPUs, SLURM scheduler |
| Hosting | Railway (Hobby) | $5/month, always on |

---

## Custom Transformer Model

### Architecture (UPDATED)

| Parameter | v4 (Broken) | v5 (Planned) |
|-----------|-------------|--------------|
| Type | Decoder-only | Decoder-only |
| Parameters | ~1.2M | **50M-100M** |
| Vocabulary size | 16,000 | 16,000 |
| Context length | 512 tokens | 512 tokens |
| Layers | 4 | TBD |
| Attention heads | 4 | TBD |
| Embedding dimension | 64 | TBD |
| Training format | Plain Q&A | **Context-aware** |

### Training Status

| Version | Status | Issue |
|---------|--------|-------|
| v1 | ❌ Abandoned | ChatGPT garbage data |
| v3 | ❌ Abandoned | Plain format |
| v4 | ❌ Abandoned | Plain format, cannot use context |
| v5 | ⬜ Pending | Context-aware, 50M-100M params |

### Training Data (Context-Format)

| Source | Rows | Context Type |
|--------|------|--------------|
| cot_qa_context.csv | 2,998 | Chain-of-thought reasoning |
| stackexchange_qa_context.csv | 10,673 | Tags + question body |
| claude_qa_batch1-38_context.csv | 15,000 | Template-based |
| **Total** | **28,671** | |

### Training Results (v1-v4, Pre-Redesign)

| Version | Data | Perplexity | Eval Score | Problem |
|---------|------|------------|------------|---------|
| v1 | 96K (garbage) | 15.55 | 14.8% | Garbage data |
| v3 | 24K (clean) | 89.63 | 16.4% | Plain format |
| v4 | 26,764 | 91.80 | 11.4% | Plain format |

### Model Files

**v4 files (OUTDATED - cannot use context):**
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

The custom model has limited knowledge capacity. RAG retrieves relevant Q&A pairs at query time as context.

**Critical requirement:** Model must be trained on context-aware format to actually USE the retrieved content.

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
| Pass rate | 94% |
| Failures | 6 (data gaps, semantic edge cases) |

---

## Inference Pipeline (Updated)

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
   
   Context:
   Q: [retrieved pair 1]
   A: [answer 1]
   
   Q: [retrieved pair 2]
   A: [answer 2]
   
   Question: [user question]
   Answer:
              │
              ▼
5. Run inference: Custom transformer (50M-100M params)
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

50M-100M parameter model is ~100-200MB. Still fits Railway.

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
| Groq fallback | User requirement: custom model must work |

---

## Next Steps

1. **Update model.py** for 50M-100M parameters
2. **Generate context-format training data** from 28,671 Q&A pairs
3. **Retrain on HPC** (30-60 min estimated)
4. **Test coherent generation** with context
5. **Connect RAG to model** for end-to-end testing
6. **Deploy to Railway**

---

*Document version: 10.0*
*Last updated: December 24, 2025*
