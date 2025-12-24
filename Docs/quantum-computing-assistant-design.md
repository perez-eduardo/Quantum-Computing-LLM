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
| Database | Neon PostgreSQL + pgvector | Free tier, auto-wake from cold start |
| Custom LLM | Trained transformer (~1.2M params) | Decoder-only, Chinchilla-optimal |
| Training Data | Claude Q&A + Stack Exchange + CoT Dataset + Books | ~5.9M tokens |
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

### Training Status

> **🔄 PHASE 1 REDO (December 24, 2025)**
> 
> - New data source identified: CoT Reasoning Dataset (3,000 pairs)
> - Retraining with expanded dataset required
> - Target: Model v4 with ~27K Q&A pairs

### Training Data

**v4 Dataset (Planned):**

| Source | Count | Est. Tokens | Status |
|--------|-------|-------------|--------|
| Claude Q&A | 15,000 pairs | ~2.3M | ✅ Ready |
| Stack Exchange (filtered) | 9,019 pairs | ~1.2M | ✅ Ready |
| **CoT Reasoning Dataset** | **3,000 pairs** | **~1.5M** | ✅ Ready |
| Books | 633,562 words | ~0.9M | ✅ Ready |
| **Total** | **~27,019 Q&A** | **~5.9M** | ⬜ Combine |

### CoT Reasoning Dataset

| Property | Value |
|----------|-------|
| Location | `data/raw/source/CoT_Reasoning_Quantum_Physics_And_Computing.json` |
| Total entries | 3,000 Q&A pairs |
| Answer length | ~3,000-4,000 chars each |
| Structure | question, answer, metadata (topic, difficulty, reasoning) |
| License | MIT (open source) |

**Why include:**
- Chain-of-thought reasoning in answers
- Self-contained explanations
- Wide topic coverage (fundamentals to advanced)
- Pre-structured Q&A format

### Claude Q&A Generation Details

| Parameter | Value |
|-----------|-------|
| Total pairs | 15,000 |
| Batches | 38 |
| Unique questions | 100% |
| Phase 1 (topics) | Batches 1-25 |
| Phase 2 (formats) | Batches 26-38 |

**Topics Covered:**
- Fundamentals, Algorithms, Hardware, Error Correction
- Chemistry, ML, Cryptography, Complexity
- Quantum Optics, Superconducting, Communication
- Trapped Ion, Neutral Atom, Photonic

**Question Formats:**
- How/Why questions
- What-if, Troubleshooting
- Comparisons, Recommendations
- Definitions, Fact-checking

### Tokenizer

**Approach:** Custom BPE tokenizer (16K vocab) trained on clean corpus.

| Token | ID | Purpose |
|-------|----|---------|
| `<pad>` | 0 | Padding for batching |
| `<eos>` | 1 | End of sequence |
| `<unk>` | 2 | Unknown token fallback |

### Training Infrastructure

| Resource | Details |
|----------|---------|
| HPC | Oregon State University |
| GPU | NVIDIA H100 80GB |
| Duration | ~13 minutes for 10 epochs (v3) |
| Throughput | ~620K tokens/sec |

---

## RAG System

### Purpose

The custom 1.2M parameter model has limited knowledge capacity. RAG compensates by retrieving relevant document chunks at query time.

**Key insight from training:** Small models learn vocabulary, not reasoning. The model outputs quantum terminology but answers are incoherent. RAG provides the actual knowledge at inference time.

### Chunking Strategy

| Parameter | Value |
|-----------|-------|
| Chunk size | ~500 tokens |
| Overlap | ~50-100 tokens |
| Metadata | Source, section, chunk index |

### Embedding Configuration

| Parameter | Value |
|-----------|-------|
| Model | voyage-3.5-lite |
| Dimensions | 1024 |
| input_type (documents) | "document" |
| input_type (queries) | "query" |

### Search Strategy

| Method | Details |
|--------|---------|
| Primary | Semantic search (pgvector cosine similarity) |
| Top-k | 5 chunks |

### RAG Data Sources

| Source | Type | Count/Size |
|--------|------|------------|
| **Books** | Chunked text | 5 books, ~2,847 chunks |
| **Stack Exchange** | Q&A pairs | 9,019 pairs |
| **CoT Reasoning Dataset** | Q&A pairs | 3,000 pairs |

#### Books

| Book | Author |
|------|--------|
| Introduction to Classical and Quantum Computing | Wong |
| Quantum Computing for Everyone | Bernhardt |
| Quantum Computing Explained for Beginners | Pantheon Space Academy |
| Quantum Computation and Quantum Information | Nielsen & Chuang |
| Quantum Computing: An Applied Approach | Hidary |

**Dual purpose:** Books used for both training (teaches vocabulary) and RAG (provides knowledge at inference).

---

## Inference Pipeline

```
1. User sends question
              │
              ▼
2. Generate query embedding (Voyage AI, input_type="query")
              │
              ▼
3. Retrieve top-k relevant chunks (pgvector)
              │
              ▼
4. Build prompt with context
              │
              ▼
5. Run inference: Custom transformer
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
      "document": "Introduction to Quantum Computing",
      "section": "Chapter 3: Entanglement"
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
| MODEL_PATH | Path to custom model weights |

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
| Fallback LLM | Custom model is the portfolio piece |

---

*Document version: 8.0*
*Last updated: December 24, 2025*
