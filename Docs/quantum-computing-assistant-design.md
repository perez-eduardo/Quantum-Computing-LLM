# Quantum Computing Assistant - Design Document

## Overview

Design specification for a web-based application that answers questions about foundational quantum computing concepts, powered by a custom-trained transformer model combined with RAG retrieval.

**Related Documents:**
- Infrastructure Planning: `initial-exploratory-brainstorming.md`
- Implementation Plan: `implementation-plan.md`

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
| Training Data | Claude Q&A + Stack Exchange + Books | ~5M tokens |
| Embeddings | Voyage AI | voyage-3.5-lite, 200M free tokens |
| Training Compute | Oregon State HPC | H100 GPUs, SLURM scheduler |
| Hosting | Railway (Hobby) | $5/month, always on |

---

## Project Directory Structure

```
E:\Personal_projects\Quantum-Computing-LLM\
│
├── venv/                           # Virtual environment
│
├── docs/                           # Documentation
│   ├── quantum-computing-assistant-design.md
│   ├── implementation-plan.md
│   └── initial-exploratory-brainstorming.md
│
├── data/
│   ├── raw/                        # Original and cleaned data
│   │   ├── chatgpt_qa_cleaned.csv      # Intermediate (template prefixes removed)
│   │   ├── chatgpt_qa_final.csv        # Final ChatGPT data (85,643 pairs)
│   │   ├── stackexchange_qa.csv        # Original Stack Exchange
│   │   ├── stackexchange_qa_cleaned.csv # Cleaned Stack Exchange (10,662 pairs)
│   │   ├── combined_qa_final.csv       # All Q&A combined (96,305 pairs)
│   │   └── books/
│   │       ├── pdf/                    # Original PDFs
│   │       │   ├── Introduction to Classical and Quantum Computing.pdf
│   │       │   ├── Quantum Computation and Quantum Information - 10th Anniversary Edition.pdf
│   │       │   ├── Quantum Computing An Applied Approach, Second edition.pdf
│   │       │   ├── Quantum Computing Explained for Beginners.pdf
│   │       │   └── Quantum Computing for Everyone.pdf
│   │       ├── beginners.txt           # Raw extracted text
│   │       ├── bernhardt.txt
│   │       ├── hidary.txt
│   │       ├── nielsen_chuang.txt
│   │       ├── wong.txt
│   │       └── cleaned/                # Cleaned book texts
│   │           ├── beginners_cleaned.txt
│   │           ├── bernhardt_cleaned.txt
│   │           ├── hidary_cleaned.txt
│   │           ├── nielsen_chuang_cleaned.txt
│   │           ├── wong_cleaned.txt
│   │           └── combined_books.txt  # All books combined (633K words)
│   └── processed/                  # Generated artifacts (RAG chunks later)
│
├── training/
│   ├── scripts/                    # Training and data processing scripts
│   │   ├── train_tokenizer.py
│   │   ├── inspect_data.py
│   │   ├── inspect_stackexchange.py
│   │   ├── inspect_book.py
│   │   ├── clean_stackexchange.py
│   │   ├── clean_book.py
│   │   ├── clean_chatgpt_qa.py
│   │   ├── clean_chatgpt_final.py
│   │   ├── combine_qa.py
│   │   └── extract_books.py
│   ├── tokenizer/                  # Tokenizer output
│   │   └── tokenizer.json          # Trained BPE tokenizer (16K vocab)
│   └── model/                      # Model weights (after training)
│
├── backend/                        # FastAPI app (Phase 3)
│
└── frontend/                       # Single HTML page (Phase 4)
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Query                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Single HTML Frontend                      │
│                   (hosted on Railway)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│       ┌─────────────┐         ┌─────────────┐               │
│       │ RAG Pipeline │         │ Custom Model │               │
│       │ (Voyage AI)  │         │ (~1.2M)      │               │
│       └─────────────┘         └─────────────┘               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Neon PostgreSQL + pgvector                      │
│                  (document chunks)                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Custom Transformer Model

### Architecture

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Type | Decoder-only (GPT-style) | Standard for text generation |
| Parameters | ~1.2M | Chinchilla-optimal for ~15M tokens |
| Vocabulary size | 16,000 | Custom BPE trained on corpus |
| Context length | 512 tokens | Sufficient for Q&A format |
| Layers | 4 | Scaled for 1.2M params |
| Attention heads | 4 | 16 dim per head |
| Embedding dimension | 64 | Scaled for 1.2M params |
| Token:param ratio | 12.5:1 | Near Chinchilla optimal (20:1) |

### Training Data Sources

> **⚠️ MAJOR REVISION (December 22, 2025)**
> 
> ChatGPT synthetic Q&A data has been **abandoned**. Investigation revealed:
> - 83% contained repetitive boilerplate phrases
> - 59% were templated questions (only numbers changed)
> - After all cleaning: 85,643 → 4,808 rows (94% garbage)
> 
> **Decision:** Replace with Claude-generated Q&A via chat sessions.

**Final Planned Data:**

| Source | Count | Type | Status |
|--------|-------|------|--------|
| Claude Q&A (planned) | ~3,000 pairs | Semi-synthetic Q&A | ⬜ Pending |
| Stack Exchange QC | 8,858 pairs | Real Q&A | ✅ Cleaned |
| 5 Books (3x upsampled) | 11,493 chunks | Text | ✅ Ready |
| **Total** | **~23,351** | | |

**Claude Q&A Generation Plan:**

| Parameter | Value |
|-----------|-------|
| Target | 3,000 beginner Q&A pairs |
| Source material | 5 quantum computing textbooks |
| Method | Semi-automated via Claude.ai chat |
| Sessions | ~20 sessions (150 Q&A each) |
| Difficulty | Beginner level |
| Cost | $0 (Pro subscription) |

**Books:**

| Book | Author | Cleaned Words |
|------|--------|---------------|
| Introduction to Classical and Quantum Computing | Thomas G. Wong | 92,947 |
| Quantum Computing for Everyone | Chris Bernhardt | 60,058 |
| Quantum Computing Explained for Beginners | Pantheon Space Academy | 64,619 |
| Quantum Computation and Quantum Information (10th Anniversary Ed.) | Nielsen & Chuang | 304,281 |
| Quantum Computing: An Applied Approach (2nd ed.) | Jack D. Hidary | 111,642 |

**Data Totals:**

| Source | Estimated Tokens |
|--------|------------------|
| Q&A pairs (~11,858) | ~1.8M |
| Books 3x upsampled (11,493 chunks) | ~2.4M |
| **Total** | **~4-5M** |

**Note:** Token count reduced from original ~15M estimate after removing ChatGPT garbage data. Model may need architecture review for Chinchilla-optimal sizing.

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

#### Efficient Domain Tokens (learned naturally)

These terms became efficient single tokens during training:
- `qubit` (ID 246)
- `qubits` (ID 287)
- `superposition` (ID 777)
- `entanglement` (ID 1649)
- `Hadamard` (ID 1756)
- `CNOT` (ID 1502)
- `gate` (ID 430)

### HPC Environment

| Resource | Details |
|----------|---------|
| Host | submit-b.hpc.engr.oregonstate.edu |
| Username | pereze4 |
| Scheduler | SLURM |
| Recommended partition | dgxh (16x H100-40GB per node) |
| Alternative partition | dgx2 (16x V100, 7-day limit) |
| Storage | 1.5 TB HPC share quota |

### Evaluation

#### Multi-Tier Approach

| Tier | Metric | Purpose |
|------|--------|---------|
| Training | Validation loss + perplexity | Monitor training, catch overfitting |
| Quality | QC test set (50-100 questions) | Measure actual capability |
| Portfolio | Examples + honest limitations | Demonstrate understanding |

#### QC Test Set

Create 50-100 test questions covering core concepts:

| Category | Example Question | Expected Keywords |
|----------|------------------|-------------------|
| Basics | What is a qubit? | "quantum bit", "superposition", "0 and 1" |
| Basics | How is a qubit different from a bit? | "superposition", "both states" |
| Entanglement | What is entanglement? | "correlated", "measuring one affects other" |
| Gates | What does a Hadamard gate do? | "superposition", "equal probability" |
| Applications | Why is quantum computing useful? | "certain problems faster", "simulation", "cryptography" |

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

RAG uses books for retrieval (cleaner, more coherent chunks than Q&A pairs):

- Introduction to Classical and Quantum Computing (Wong)
- Quantum Computing for Everyone (Bernhardt)
- Quantum Computing Explained for Beginners
- Quantum Computation and Quantum Information (Nielsen & Chuang)
- Quantum Computing: An Applied Approach (Hidary)

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

### Error Handling

| Scenario | Response |
|----------|----------|
| Model inference failure | 500 + error message |
| Database connection error | 503 + "Service temporarily unavailable" |
| No relevant chunks found | 200 + answer with disclaimer |
| Rate limit hit | 429 + "Rate limit exceeded" |

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

### Wireframe

```
┌─────────────────────────────────────────────────────────────┐
│  Quantum Computing Assistant                                 │
│  Powered by a custom-trained 1.2M parameter transformer      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Ask a question about quantum computing...              │ │
│  └────────────────────────────────────────────────────────┘ │
│                                              [ Ask ]         │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Answer:                                                     │
│  ─────────────────────────────────────────────────────────  │
│  [Response appears here]                                     │
│                                                              │
│  Sources: [expandable]                                       │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│  For educational purposes only. | GitHub | Portfolio         │
└─────────────────────────────────────────────────────────────┘
```

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

*Document version: 3.0*
*Last updated: December 22, 2025*
