# Quantum Computing Assistant - Implementation Plan

**Last Updated:** December 23, 2025

**Related Documents:**
- Design Document: `quantum-computing-assistant-design.md`
- Infrastructure Planning: `initial-exploratory-brainstorming.md`
- Model Investigation Report: `model_investigation_report.md`

---

## Current Status

**Phase:** 1 (Training Pipeline) - Data generation complete, ready for retraining
**Status:** Claude Q&A generation complete (15,000 pairs). Ready to combine dataset and retrain.

---

## What Has Been Done

### Data Acquisition

| Task | Status | Notes |
|------|--------|-------|
| Download QuantumLLMInstruct | ✅ Done | Only 5.1K pairs available (not 500K as advertised). Only 46 usable. Excluded from final data. |
| Download Stack Exchange QC dump | ✅ Done | 28K total posts (13K questions, 15K answers) |
| ~~Obtain ChatGPT synthetic Q&A~~ | ❌ Abandoned | 94% garbage (boilerplate + templates). Replaced with Claude Q&A. |
| Obtain book PDFs | ✅ Done | 5 books |
| **Generate Claude Q&A** | ✅ Done | **15,000 pairs across 38 batches** |

### Data Processing & Cleaning

| Task | Status | Output |
|------|--------|--------|
| Process Stack Exchange XML | ✅ Done | `stackexchange_qa.csv` (10,673 pairs) |
| Clean Stack Exchange (HTML entities, short answers) | ✅ Done | `stackexchange_qa_cleaned.csv` (10,662 pairs) |
| Filter Stack Exchange (>1024 tokens) | ✅ Done | `stackexchange_qa_cleaned.csv` (8,858 pairs) |
| Extract text from PDFs | ✅ Done | 5 text files |
| Clean book texts (remove fragments, TOC, etc.) | ✅ Done | `books/cleaned/` (633,562 words total) |
| Upsample book data 3x | ✅ Done | 11,493 chunks |
| ~~Clean ChatGPT data~~ | ❌ Abandoned | Data unsalvageable |
| **Generate Claude Q&A (38 batches)** | ✅ Done | `claude_qa_batch[1-38].csv` (15,000 pairs) |
| Combine Q&A sources | ⬜ Pending | Target: ~35,351 examples |
| Combine book texts | ✅ Done | `books/cleaned/combined_books.txt` |
| Train custom BPE tokenizer | ✅ Done | `tokenizer/tokenizer.json` (16K vocab) |
| **Retrain tokenizer on new corpus** | ⬜ Pending | After combining new data |

### Claude Q&A Generation Details

| Metric | Value |
|--------|-------|
| Total pairs generated | **15,000** |
| Batches | 38 (25 × 400 + 12 × 400 + 1 × 200) |
| Unique questions | 15,000 (100%) |
| Duplicates fixed during generation | ~250 |
| Verification method | 8-chunk per batch with index checking |

**Topics Covered:**

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
| 26 | "What/How does X work" (mechanisms) |
| 27 | "Why" questions (reasoning and causes) |
| 28 | "What happens when/if" (consequences and scenarios) |
| 29 | Troubleshooting and problem-solving |
| 30 | Best practices and recommendations |
| 31 | Definitions and explanations |
| 32 | "When/What if" conditional questions |
| 33 | "Can/Could" possibility questions |
| 34 | "Is it true that" fact-checking questions |
| 35 | "Which" choice and selection questions |
| 36 | Comparison questions (X vs Y) |
| 37 | "Should I" recommendation questions |
| 38 | Mixed final questions |

### HPC Training

| Task | Status | Notes |
|------|--------|-------|
| Set up HPC environment | ✅ Done | Python 3.11 venv, PyTorch 2.5.1+cu121 |
| Implement transformer architecture | ✅ Done | `scripts/model.py` |
| Write dataset loader | ✅ Done | `scripts/dataset.py` |
| Write training script | ✅ Done | `scripts/train.py` |
| Create SLURM job script | ✅ Done | `train_job.sh` |
| Train model on H100 (initial) | ✅ Done | 4 minutes, 3 epochs |
| Evaluate model quality | ✅ Done | 14.8% keyword match, incoherent outputs |
| Investigate data quality issues | ✅ Done | Found ChatGPT data was 94% garbage |
| **Retrain with new data** | ⬜ Pending | 10 epochs planned |

### Model Investigation

| Task | Status | Findings |
|------|--------|----------|
| Evaluate with 50-question test set | ✅ Done | 14.8% keyword match |
| Compare checkpoints (epoch 1, 3, 6) | ✅ Done | Epoch 6 >> 3 >> 1, more training helps |
| Analyze token length distribution | ✅ Done | 88% under 128 tokens, 5.5% truncated |
| Inspect boilerplate in ChatGPT data | ✅ Done | 83.4% contained boilerplate phrases |
| Inspect templated examples | ✅ Done | 59% were templated garbage |
| **Decision: Abandon ChatGPT data** | ✅ Done | Replaced with Claude Q&A |

---

## Training Results (Initial Run - Before Data Fix)

**Job:** 19739587 on dgxh-1 (H100 80GB)
**Duration:** ~4 minutes
**Throughput:** ~626K tokens/sec

| Epoch | Train Loss | Val Loss | Perplexity |
|-------|------------|----------|------------|
| 1 | 5.32 | 3.23 | 25.29 |
| 2 | 2.94 | 2.82 | 16.84 |
| 3 | 2.74 | 2.74 | 15.55 |

**Issues Identified:**
- Loss still decreasing at epoch 3 (undertrained)
- Model memorized boilerplate from ChatGPT data
- Only 4,461 steps total

---

## Final Dataset Composition

| Source | Count | Est. Tokens | Status |
|--------|-------|-------------|--------|
| Claude Q&A | 15,000 pairs | ~2.3M | ✅ Complete |
| Stack Exchange (cleaned) | 8,858 pairs | ~1.2M | ✅ Ready |
| Books (3x upsampled) | 11,493 chunks | ~2.4M | ✅ Ready |
| **Total** | **~35,351** | **~5.9M** | |

**Note on books:** Books serve dual purpose in this project:
- **Training:** CLM chunks (3x upsampled with offset) teach vocabulary and patterns
- **RAG (Phase 2):** Semantic chunks (~500 tokens with overlap) provide retrieval at inference

This is complementary, not redundant. Training teaches the model how to speak; RAG injects specific facts the small model cannot memorize.

---

## What Is Next

**Immediate next task:** Combine dataset, retrain tokenizer, retrain model

### Remaining Phase 1 Tasks

| Task | Priority | Status |
|------|----------|--------|
| Generate Claude Q&A | - | ✅ Done (15,000 pairs) |
| Combine final dataset | **Next** | ⬜ Pending |
| Retrain tokenizer on new corpus | **Next** | ⬜ Pending |
| Retrain with cleaned data (10 epochs) | After combining | ⬜ Pending |
| Re-evaluate model | After retraining | ⬜ Pending |
| Document results for portfolio | After evaluation | ⬜ Pending |

### Planned Retraining Command

```bash
~/hpc-share/quantum-llm/venv/bin/python scripts/train.py \
    --epochs 10 \
    --min_lr 1e-5 \
    --warmup_ratio 0.05
```

---

## Output Files Created

### Data Files

| File | Location | Description | Size |
|------|----------|-------------|------|
| `claude_qa_batch[1-25].csv` | `data/raw/` | Claude Q&A Phase 1 (10,000 pairs) | ~3 MB |
| `claude_qa_batch[26-38].csv` | Generated | Claude Q&A Phase 2 (5,000 pairs) | ~1.5 MB |
| `questions_index.txt` | Generated | Duplicate detection index | 15,000 q |
| `stackexchange_qa_cleaned.csv` | `data/raw/` | Stack Exchange Q&A (cleaned) | ~24 MB |
| `combined_books.txt` | HPC `data/` | All book text (cleaned) | ~3.8 MB |

### Training Artifacts (HPC: ~/hpc-share/quantum-llm/)

| File | Location | Description | Size |
|------|----------|-------------|------|
| `final_model.pt` | `model/` | Final model weights (epoch 3) | 5 MB |
| `checkpoint_epoch1.pt` | `model/` | Epoch 1 checkpoint | 15 MB |
| `checkpoint_epoch2.pt` | `model/` | Epoch 2 checkpoint | 15 MB |
| `checkpoint_epoch3.pt` | `model/` | Epoch 3 checkpoint | 15 MB |
| `checkpoint_epoch6.pt` | `model/` | Epoch 6 checkpoint (from cancelled job) | 15 MB |
| `config.json` | `model/` | Model config | 149 B |
| `tokenizer.json` | `/` | Tokenizer (needs retraining) | - |

---

## Development Phases

### Phase 1: Training Pipeline

Build and train the custom transformer model.

| Task | Status |
|------|--------|
| Decide on training data sources | ✅ Done |
| Define model size (1.2M params) | ✅ Done |
| Define tokenizer approach (custom BPE, 16K vocab) | ✅ Done |
| Define evaluation metrics | ✅ Done |
| Download QuantumLLMInstruct | ✅ Done |
| Download Stack Exchange data dump | ✅ Done |
| ~~Load ChatGPT synthetic Q&A~~ | ❌ Abandoned |
| **Generate Claude Q&A (15,000 pairs)** | ✅ Done |
| Extract text from book PDFs | ✅ Done |
| Preprocess and combine all sources | 🔄 In Progress |
| Clean and verify all data | ✅ Done |
| Train custom tokenizer | ✅ Done (needs retrain) |
| Set up HPC environment | ✅ Done |
| Implement transformer architecture | ✅ Done |
| Write training script | ✅ Done |
| Train model on HPC | ✅ Done (needs retrain) |
| Evaluate model | ✅ Done |
| Investigate data quality issues | ✅ Done |
| **Retrain model with clean data** | ⬜ Pending |
| **Re-evaluate model** | ⬜ Pending |
| Document results | ⬜ Pending |

### Phase 2: RAG System

Build retrieval-augmented generation pipeline.

| Task | Status |
|------|--------|
| Chunk books for RAG (~500 tokens, semantic) | ⬜ Pending |
| Generate embeddings (Voyage AI) | ⬜ Pending |
| Set up Neon database with pgvector | ⬜ Pending |
| Implement retrieval pipeline | ⬜ Pending |
| Test retrieval quality | ⬜ Pending |

**Note:** RAG uses different chunking than training. Training used CLM chunks (3x upsampled). RAG needs ~500 token semantic chunks with overlap for retrieval.

### Phase 3: Backend

Build FastAPI backend.

| Task | Status |
|------|--------|
| FastAPI endpoints | ⬜ Pending |
| Custom model inference | ⬜ Pending |
| RAG integration | ⬜ Pending |
| Error handling | ⬜ Pending |
| Rate limiting | ⬜ Pending |

### Phase 4: Frontend

Build single-page UI.

| Task | Status |
|------|--------|
| Single HTML page | ⬜ Pending |
| API integration | ⬜ Pending |
| Response display with sources | ⬜ Pending |
| Basic styling | ⬜ Pending |

### Phase 5: Deployment

Deploy to Railway.

| Task | Status |
|------|--------|
| Deploy to Railway | ⬜ Pending |
| Configure environment variables | ⬜ Pending |
| Test end-to-end | ⬜ Pending |
| Set spending caps | ⬜ Pending |

---

## Key Findings During Implementation

1. **QuantumLLMInstruct dataset is incomplete.** Advertised as 500K pairs, only ~5K available publicly. Of those, only 46 have complete problem-solution pairs. Excluded from final training data.

2. **ChatGPT synthetic data was 94% garbage.** Boilerplate phrases in 83%, templated questions in 59%. Abandoned entirely and replaced with Claude Q&A.

3. **PDF extraction produces garbage.** Raw text extraction creates fragments from math formulas, TOC dots, page numbers. Required paragraph-level filtering to remove fragments <50 chars.

4. **Stack Exchange data was mostly clean.** Only needed filtering for truncated examples (>1024 tokens).

5. **Chinchilla scaling matters.** Original 10M param target was too large for ~15M tokens. Reduced to 1.2M params for ~12.5:1 token-to-param ratio.

6. **Inspect after every step.** Early pipeline ran garbage through tokenizer. Proper workflow: acquire → inspect → clean → verify → proceed.

7. **Test locally before submitting HPC jobs.** Wasted 2 job submissions due to missing dependencies.

8. **H100s are fast.** 1.2M param model trained in 4 minutes at 626K tokens/sec.

9. **Small models learn vocabulary, not reasoning.** Model outputs quantum jargon but answers are incoherent. RAG is essential for usability.

10. **Chunk-based generation with verification works.** Generating Q&A in 8-chunk batches with incremental duplicate checking caught issues early and scaled to 15,000 pairs.

11. **More epochs help small models.** Checkpoint comparison showed epoch 6 >> epoch 3 >> epoch 1. Plan to use 10 epochs for retraining.

12. **Diverse question formats improve coverage.** Phase 2 added question types (how, why, what-if, should-I, comparisons) to complement Phase 1 topic coverage.

---

## Notes for Next Session

- Claude Q&A generation complete: 15,000 pairs in `claude_qa_batch[1-38].csv`
- Need to combine with Stack Exchange (8,858) and books (11,493 chunks)
- Retrain tokenizer on new corpus before model training
- Train with 10 epochs, min_lr 1e-5, warmup 0.05
- Expected dataset size: ~35,351 examples, ~5.9M tokens

---

*Document version: 7.0*
