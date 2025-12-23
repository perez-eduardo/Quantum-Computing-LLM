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
| Training Data | Claude Q&A + Stack Exchange + Books | ~5.9M tokens |
| Embeddings | Voyage AI | voyage-3.5-lite, 200M free tokens |
| Training Compute | Oregon State HPC | H100 GPUs, SLURM scheduler |
| Hosting | Railway (Hobby) | $5/month, always on |

---

## Custom Transformer Model

### Architecture

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Type | Decoder-only (GPT-style) | Standard for text generation |
| Parameters | ~1.2M | Chinchilla-optimal for ~5.9M tokens |
| Vocabulary size | 16,000 | Custom BPE trained on corpus |
| Context length | 512 tokens | Sufficient for Q&A format |
| Layers | 4 | Scaled for 1.2M params |
| Attention heads | 4 | 16 dim per head |
| Embedding dimension | 64 | Scaled for 1.2M params |
| Token:param ratio | ~5:1 | Conservative for quality |

### Training Data Sources

> **✅ DATA GENERATION COMPLETE (December 23, 2025)**
> 
> - ChatGPT synthetic data was **abandoned** (94% garbage)
> - Claude Q&A generation **complete**: 15,000 pairs across 38 batches
> - Ready for dataset assembly and retraining

**Final Dataset:**

| Source | Count | Est. Tokens | Status |
|--------|-------|-------------|--------|
| Claude Q&A | 15,000 pairs | ~2.3M | ✅ Complete |
| Stack Exchange QC | 8,858 pairs | ~1.2M | ✅ Cleaned |
| 5 Books (3x upsampled) | 11,493 chunks | ~2.4M | ✅ Ready |
| **Total** | **~35,351** | **~5.9M** | |

### Claude Q&A Generation Details

| Parameter | Value |
|-----------|-------|
| Total pairs | **15,000** |
| Batches | 38 (Phase 1: 25 × 400, Phase 2: 12 × 400 + 1 × 200) |
| Unique questions | 15,000 (100%) |
| Verification | 8-chunk per batch with index checking |
| Output files | `claude_qa_batch[1-38].csv` |

**Topics Covered (38 Batches):**

**Phase 1 - Topic-Based (Batches 1-25):**
| Batches | Topics |
|---------|--------|
| 1-5 | Fundamentals, Algorithms, Hardware, Error Correction, Chemistry |
| 6-10 | ML, Cryptography, Complexity, Many-body, Topological |
| 11-15 | Simulation, Annealing, Control, Metrology, Sensing |
| 16-20 | Thermodynamics, Foundations, Optics, Superconducting, Communication |
| 21-22 | Trapped Ion, Neutral Atom |
| 23 | Photonic Quantum Computing |
| 24 | Quantum Software and Programming |
| 25 | Quantum Applications and Industry |

**Phase 2 - Question Format-Based (Batches 26-38):**
| Batches | Question Formats |
|---------|------------------|
| 26-27 | How/Why questions |
| 28-29 | What-if, Troubleshooting |
| 30-31 | Best practices, Definitions |
| 32-33 | Conditional, Possibility questions |
| 34-35 | Fact-checking, Choice questions |
| 36-37 | Comparisons, Recommendations |
| 38 | Mixed final questions |

**Books:**

| Book | Author | Cleaned Words |
|------|--------|---------------|
| Introduction to Classical and Quantum Computing | Thomas G. Wong | 92,947 |
| Quantum Computing for Everyone | Chris Bernhardt | 60,058 |
| Quantum Computing Explained for Beginners | Pantheon Space Academy | 64,619 |
| Quantum Computation and Quantum Information (10th Anniversary Ed.) | Nielsen & Chuang | 304,281 |
| Quantum Computing: An Applied Approach (2nd ed.) | Jack D. Hidary | 111,642 |

**Licensing:**
- Claude Q&A: Self-generated via Pro subscription
- Stack Exchange: Personal use only, no LLM distribution
- Books: Educational/personal use only

Model weights kept private.

### Tokenizer

**Approach:** Custom BPE tokenizer (16K vocab) trained on the corpus.

Why custom over GPT-2 tokenizer:
- Shows complete "from scratch" pipeline (portfolio value)
- Optimized for QC terminology
- Smaller vocab (16K) matches smaller model

#### Special Tokens

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
| Partition | dgxh (4-hour limit) |
| Alternative partition | dgx2 (16x V100, 7-day limit) |
| Storage | 1.5 TB HPC share quota |

### Evaluation

#### Multi-Tier Approach

| Tier | Metric | Purpose |
|------|--------|---------|
| Training | Validation loss + perplexity | Monitor training, catch overfitting |
| Quality | QC test set (50-100 questions) | Measure actual capability |
| Portfolio | Examples + honest limitations | Demonstrate understanding |

---

## RAG System

### Purpose

The custom 1.2M parameter model has limited knowledge capacity. RAG compensates by retrieving relevant document chunks at query time, providing context the small model couldn't memorize.

### Chunking Strategy

| Parameter | Value |
|-----------|-------|
| Chunk size | ~500 tokens |
| Overlap | ~50-100 tokens |
| Metadata | Source document, section, chunk index |

### Search Strategy

| Method | Details |
|--------|---------|
| Primary | Semantic search (pgvector cosine similarity) |
| Top-k | 5 chunks |
| Reranking | TBD (may not be needed for MVP) |

### Data Sources

RAG retrieves from the same 5 books used in training, but with different chunking:

| Book | Author |
|------|--------|
| Introduction to Classical and Quantum Computing | Wong |
| Quantum Computing for Everyone | Bernhardt |
| Quantum Computing Explained for Beginners | Pantheon Space Academy |
| Quantum Computation and Quantum Information | Nielsen & Chuang |
| Quantum Computing: An Applied Approach | Hidary |

**Why books serve dual purpose (training + RAG):**

| Use | Chunking Strategy | Purpose |
|-----|-------------------|---------|
| Training | CLM chunks, 3x upsampled with offset | Teaches vocabulary and patterns |
| RAG | ~500 token semantic chunks with overlap | Provides specific facts at query time |

This is not redundant. Training and RAG serve complementary roles: training teaches the model how to speak about quantum computing, while RAG injects specific knowledge the small model couldn't memorize.

---

## Inference Pipeline

### Query Flow

```
1. User sends question
              │
              ▼
2. Generate query embedding (Voyage AI)
              │
              ▼
3. Retrieve top-k relevant chunks (pgvector)
              │
              ▼
4. Build prompt:
   - System: "You are a quantum computing assistant..."
   - Context: Retrieved chunks
   - User: Current question
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

POST /query Request:
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
      "section": "Chapter 3: Entanglement",
      "excerpt": "..."
    }
  ]
}
```

---

## UI Design

### Layout

Single HTML page with minimal design. Self-contained (no framework dependencies).

| Element | Description |
|---------|-------------|
| Header | Title, brief description |
| Input area | Text input for questions |
| Response area | Model's answer with sources |
| Footer | Disclaimer, portfolio link |

### Style Direction

- Clean, minimal, professional
- Light mode
- Monospace font for technical terms
- No external CSS frameworks (vanilla CSS)

---

## Hosting & Deployment

### Railway Configuration

| Setting | Value |
|---------|-------|
| Plan | Hobby ($5/month) |
| Structure | Monorepo (backend serves frontend) |
| Always on | Yes |
| URL | TBD (your-app.up.railway.app) |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| VOYAGE_API_KEY | Embeddings |
| DATABASE_URL | Neon PostgreSQL connection |
| MODEL_PATH | Path to custom model weights |

### Model Deployment

The trained model weights (~5MB for ~1.2M params) will be:
- Stored in the Railway container
- Loaded into memory on startup
- Kept in memory (Railway always-on)

---

## Cost Summary

| Component | Monthly Cost |
|-----------|--------------|
| Railway (hosting) | $5 |
| Voyage AI (embeddings) | $0 (free tier) |
| Neon (database) | $0 (free tier) |
| HPC (training) | $0 (one-time, university resource) |
| **Total** | **$5/month** |

---

## Cost Protection

| Provider | Hard Cap | Action |
|----------|----------|--------|
| Railway | $10 | Settings > Usage > Hard limit |
| Voyage AI | N/A | 200M token ceiling |
| Neon | N/A | Auto-stops at 100 compute hours |

**Maximum exposure: $10/month**

---

## Out of Scope (MVP)

| Feature | Reason |
|---------|--------|
| Caching | Low traffic, adds complexity for minimal benefit |
| Comparison mode | Not needed for portfolio demonstration |
| User level selection | 1.2M model too small to adjust tone |
| Fallback LLM | Custom model is the portfolio piece |

These can be added later if needed.

---

*Document version: 6.0*
*Last updated: December 24, 2025*
