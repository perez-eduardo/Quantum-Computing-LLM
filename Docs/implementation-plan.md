# Quantum Computing Assistant - Implementation Plan

**Last Updated:** December 24, 2025

**Related Documents:**
- Design Document: `quantum-computing-assistant-design.md`
- Infrastructure Planning: `initial-exploratory-brainstorming.md`
- Model Investigation Report: `model_investigation_report.md`

---

## Current Status

**Phase:** 1 (Training Pipeline) - Complete
**Status:** Model v3 trained on clean data. Ready for evaluation.

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
| Filter Stack Exchange (>1024 tokens) | ✅ Done | 9,019 pairs after filtering |
| Extract text from PDFs | ✅ Done | 5 text files |
| Clean book texts (remove fragments, TOC, etc.) | ✅ Done | `books/cleaned/` (633,562 words total) |
| ~~Clean ChatGPT data~~ | ❌ Abandoned | Data unsalvageable |
| **Generate Claude Q&A (38 batches)** | ✅ Done | `claude_qa.csv` (15,000 pairs) |
| Combine Q&A sources | ✅ Done | 24,019 pairs (after filtering + dedup) |
| Combine book texts | ✅ Done | `combined_books.txt` |
| Train custom BPE tokenizer | ✅ Done | `tokenizer.json` (16K vocab, clean corpus) |

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
| Create SLURM job script | ✅ Done | `train_job_v3.sh` |
| Train model v1 (garbage data) | ✅ Done | 3 epochs, perplexity 15.55 |
| Investigate data quality issues | ✅ Done | Found ChatGPT data was 94% garbage |
| **Train model v3 (clean data)** | ✅ Done | **10 epochs, perplexity 89.63** |
| Evaluate model | ⬜ Pending | Next step |

### Model Investigation

| Task | Status | Findings |
|------|--------|----------|
| Evaluate with 50-question test set | ✅ Done | 14.8% keyword match (v1) |
| Compare checkpoints (epoch 1, 3, 6) | ✅ Done | Epoch 6 >> 3 >> 1, more training helps |
| Analyze token length distribution | ✅ Done | 88% under 128 tokens |
| Inspect boilerplate in ChatGPT data | ✅ Done | 83.4% contained boilerplate phrases |
| Inspect templated examples | ✅ Done | 59% were templated garbage |
| **Decision: Abandon ChatGPT data** | ✅ Done | Replaced with Claude Q&A |

---

## Training Results

### v1 (Garbage Data - December 21, 2025)

**Job:** 19739587 on dgxh-1 (H100 80GB)
**Duration:** ~4 minutes
**Data:** 96K Q&A pairs (94% ChatGPT garbage)

| Epoch | Train Loss | Val Loss | Perplexity |
|-------|------------|----------|------------|
| 1 | 5.32 | 3.23 | 25.29 |
| 2 | 2.94 | 2.82 | 16.84 |
| 3 | 2.74 | 2.74 | 15.55 |

**Issues:** Model memorized boilerplate, incoherent outputs, 14.8% eval accuracy.

### v3 (Clean Data - December 23, 2025)

**Job:** 19759979 on dgxh-1 (H100 80GB)
**Duration:** ~13 minutes
**Data:** 24,019 Q&A pairs (Claude + Stack Exchange) + books

| Epoch | Train Loss | Val Loss | Perplexity |
|-------|------------|----------|------------|
| 1 | 7.95 | 6.39 | 594.77 |
| 2 | 5.83 | 5.40 | 221.96 |
| 3 | 5.26 | 5.04 | 153.94 |
| 4 | 4.98 | 4.82 | 124.56 |
| 5 | 4.81 | 4.69 | 108.64 |
| 6 | 4.70 | 4.60 | 99.61 |
| 7 | 4.63 | 4.55 | 94.43 |
| 8 | 4.59 | 4.52 | 91.56 |
| 9 | 4.56 | 4.50 | 90.26 |
| **10** | **4.55** | **4.50** | **89.63** |

**Training Config (v3):**

| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Batch size | 64 |
| Max LR | 3e-4 |
| Min LR | 1e-5 |
| Warmup ratio | 0.05 |
| Max seq length | 512 |

**Note:** Higher perplexity than v1 is expected. v1 had low perplexity because it memorized repetitive garbage. v3 trained on diverse, clean data produces higher perplexity but should have better output quality. The real test is evaluation.

---

## Final Dataset Composition

| Source | Count | Est. Tokens | Status |
|--------|-------|-------------|--------|
| Claude Q&A | 15,000 pairs | ~2.3M | ✅ Complete |
| Stack Exchange (filtered) | 9,019 pairs | ~1.2M | ✅ Complete |
| Books | 633,562 words | ~0.9M | ✅ Complete |
| **Total** | **24,019 Q&A** | **~4.4M** | |

**Note:** 1,640 Stack Exchange examples >1024 tokens were filtered. 3 duplicate answers removed. Final Q&A count: 24,019.

**Books serve dual purpose:**
- **Training:** CLM chunks teach vocabulary and patterns
- **RAG (Phase 2):** Semantic chunks (~500 tokens with overlap) provide retrieval at inference

---

## What Is Next

**Immediate next task:** Evaluate model quality

### Remaining Phase 1 Tasks

| Task | Priority | Status |
|------|----------|--------|
| Train model v3 | - | ✅ Done |
| **Evaluate model quality** | **Next** | ⬜ Pending |
| Compare v1 vs v3 outputs | After eval | ⬜ Pending |
| Document results for portfolio | After evaluation | ⬜ Pending |

### Evaluation Plan

1. Run same 50-question test set used for v1
2. Compare keyword match scores
3. Qualitative assessment of output coherence
4. Check for boilerplate memorization

---

## Output Files Created

### Data Files (Local)

| File | Location | Description |
|------|----------|-------------|
| `claude_qa.csv` | `data/raw/` | Claude Q&A (15,000 pairs) |
| `stackexchange_qa_cleaned.csv` | `data/raw/` | Stack Exchange Q&A |
| `combined_books.txt` | `data/raw/books/` | All book text |

### Data Files (HPC)

| File | Location | Description |
|------|----------|-------------|
| `combined_qa_filtered.csv` | `data/` | Final Q&A (24,019 pairs) |
| `combined_books.txt` | `data/` | Book text for training |
| `tokenizer.json` | `/` | BPE tokenizer (16K vocab) |

### Training Artifacts (HPC: ~/hpc-share/quantum-llm/)

| File | Location | Description |
|------|----------|-------------|
| `final_model.pt` | `model/` | Final model weights (v3, epoch 10) |
| `best_model.pt` | `model/` | Best model by val loss |
| `checkpoint_epoch[1-10].pt` | `model/` | All epoch checkpoints |
| `config.json` | `model/` | Model config |
| `train_job_v3.sh` | `/` | SLURM job script |

---

## Development Phases

### Phase 1: Training Pipeline ✅

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
| Preprocess and combine all sources | ✅ Done |
| Clean and verify all data | ✅ Done |
| Train custom tokenizer | ✅ Done |
| Set up HPC environment | ✅ Done |
| Implement transformer architecture | ✅ Done |
| Write training script | ✅ Done |
| **Train model v3 (clean data)** | ✅ Done |
| Evaluate model | ⬜ Pending |
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

4. **Stack Exchange data needed filtering.** Removed 1,640 examples >1024 tokens (all from Stack Exchange, 0 from Claude).

5. **Chinchilla scaling matters.** Original 10M param target was too large for ~15M tokens. Reduced to 1.2M params for better token-to-param ratio.

6. **Inspect after every step.** Early pipeline ran garbage through tokenizer. Proper workflow: acquire → inspect → clean → verify → proceed.

7. **Test locally before submitting HPC jobs.** Wasted 2 job submissions due to missing dependencies.

8. **H100s are fast.** 1.2M param model trained 10 epochs in 13 minutes at 626K tokens/sec.

9. **Small models learn vocabulary, not reasoning.** Model outputs quantum jargon but answers are incoherent. RAG is essential for usability.

10. **Chunk-based generation with verification works.** Generating Q&A in 8-chunk batches with incremental duplicate checking caught issues early and scaled to 15,000 pairs.

11. **More epochs help small models.** Checkpoint comparison showed epoch 6 >> epoch 3 >> epoch 1. v3 used 10 epochs.

12. **Diverse question formats improve coverage.** Phase 2 added question types (how, why, what-if, should-I, comparisons) to complement Phase 1 topic coverage.

13. **Don't load system Python module after venv activation.** `module load python/3.11` overrides venv's Python and breaks imports.

14. **Retrain tokenizer when data changes.** New tokenizer trained on clean corpus only (no ChatGPT garbage).

---

## Notes for Next Session

- Model v3 training complete: 10 epochs, perplexity 89.63
- Files on HPC: `model/final_model.pt`, `model/best_model.pt`
- Next step: Run evaluation to compare v1 vs v3 quality
- Generate script or use existing eval script with 50-question test set

---

*Document version: 8.0*
