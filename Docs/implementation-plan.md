# Quantum Computing Assistant - Implementation Plan

**Last Updated:** December 21, 2025

**Related Documents:**
- Design Document: `quantum-computing-assistant-design.md`
- Infrastructure Planning: `initial-exploratory-brainstorming.md`

---

## Current Status

**Phase:** 1 (Training Pipeline)
**Status:** Model trained. QA evaluation pending.

---

## What Has Been Done

### Data Acquisition

| Task | Status | Notes |
|------|--------|-------|
| Download QuantumLLMInstruct | ✅ Done | Only 5.1K pairs available (not 500K as advertised). Only 46 usable. Excluded from final data. |
| Download Stack Exchange QC dump | ✅ Done | 28K total posts (13K questions, 15K answers) |
| Obtain ChatGPT synthetic Q&A | ✅ Done | 98K pairs (original), heavily deduplicated |
| Obtain book PDFs | ✅ Done | 5 books |

### Data Processing & Cleaning

| Task | Status | Output |
|------|--------|--------|
| Process Stack Exchange XML | ✅ Done | `stackexchange_qa.csv` (10,673 pairs) |
| Clean Stack Exchange (HTML entities, short answers) | ✅ Done | `stackexchange_qa_cleaned.csv` (10,662 pairs) |
| Extract text from PDFs | ✅ Done | 5 text files |
| Clean book texts (remove fragments, TOC, etc.) | ✅ Done | `books/cleaned/` (633,562 words total) |
| Clean ChatGPT (remove template prefixes) | ✅ Done | `chatgpt_qa_cleaned.csv` |
| Clean ChatGPT (remove short entries, deduplicate) | ✅ Done | `chatgpt_qa_final.csv` (85,643 pairs) |
| Combine Q&A sources | ✅ Done | `combined_qa_final.csv` (96,305 pairs) |
| Combine book texts | ✅ Done | `books/cleaned/combined_books.txt` |
| Train custom BPE tokenizer | ✅ Done | `tokenizer/tokenizer.json` (16K vocab) |

### HPC Training

| Task | Status | Notes |
|------|--------|-------|
| Set up HPC environment | ✅ Done | Python 3.11 venv, PyTorch 2.5.1+cu121 |
| Implement transformer architecture | ✅ Done | `scripts/model.py` |
| Write dataset loader | ✅ Done | `scripts/dataset.py` |
| Write training script | ✅ Done | `scripts/train.py` |
| Create SLURM job script | ✅ Done | `train_job.sh` |
| Train model on H100 | ✅ Done | 8 minutes, 3 epochs |

### Architecture Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| Model size | 1.2M params (1,286,720 actual) | Chinchilla-optimal for ~15M tokens |
| Embedding dim | 64 | Scaled for 1.2M params |
| Layers | 4 | Balanced depth |
| Heads | 4 | 16 dim per head |
| Vocabulary | 16,000 | Custom BPE on QC corpus |
| Context length | 512 | Sufficient for Q&A |
| Token-to-param ratio | 12.5:1 | Near optimal (20:1 is Chinchilla) |
| Positional encoding | RoPE | Modern, no learned positions |
| Normalization | RMSNorm | Efficient, stable |
| FFN activation | SwiGLU | Better than ReLU/GELU |

---

## Training Results

**Job:** 19739587 on dgxh-1 (H100 80GB)
**Duration:** ~8 minutes
**Throughput:** ~626K tokens/sec

| Epoch | Train Loss | Val Loss | Perplexity |
|-------|------------|----------|------------|
| 1 | 5.32 | 3.23 | 25.29 |
| 2 | 2.94 | 2.82 | 16.84 |
| 3 | 2.74 | 2.74 | 15.55 |

**Observations:**
- Loss decreased consistently (no overfitting)
- Final perplexity 15.55 (reasonable for 1.2M params)
- Model learned quantum terminology
- Generation is incoherent (expected for this size)

**Sample outputs:**
```
Q: What is superposition?
A: If a basis is a state |ψ⟩=λ|ψ⟩ for classical outcome...

Q: What is entanglement?
A: Pauli-Y is a NOT) and corresponds to a 180° rotation about the Z-axis...

Q: What is a Hadamard gate?
A: A pure states of a basis states (e.g., with H or S†H) can reveal phase/coherence...
```

Model produces quantum jargon but not coherent answers. RAG will compensate.

---

## Final Clean Data Summary

### Q&A Data

| Source | Original | After Cleaning | Removed |
|--------|----------|----------------|---------|
| ChatGPT Synthetic | 98,228 | 85,643 | 12,585 (duplicates, short entries) |
| Stack Exchange | 10,673 | 10,662 | 11 (HTML entities, short answers) |
| **Total** | **108,901** | **96,305** | **12,596** |

### Book Data

| Book | Original Words | Cleaned Words | Removed |
|------|----------------|---------------|---------|
| beginners.txt | 65,139 | 64,619 | 0.8% |
| bernhardt.txt | 71,481 | 60,058 | 16% |
| hidary.txt | 117,240 | 111,642 | 4.8% |
| nielsen_chuang.txt | 316,696 | 304,281 | 3.9% |
| wong.txt | 114,531 | 92,947 | 18.8% |
| **Total** | **685,087** | **633,547** | **7.5%** |

### Token Estimates

| Source | Tokens |
|--------|--------|
| Q&A pairs | ~14.2M |
| Books | ~0.98M (actual from tokenizer) |
| **Total** | **~15M** |

---

## Output Files Created

### Data Files

| File | Location | Description | Size |
|------|----------|-------------|------|
| `combined_qa_final.csv` | `data/raw/` | All Q&A pairs (cleaned) | ~56 MB |
| `chatgpt_qa_final.csv` | `data/raw/` | ChatGPT Q&A (cleaned) | ~38 MB |
| `stackexchange_qa_cleaned.csv` | `data/raw/` | Stack Exchange Q&A (cleaned) | ~24 MB |
| `combined_books.txt` | `data/raw/books/cleaned/` | All book text (cleaned) | ~3.8 MB |

### Training Artifacts (Local)

| File | Location | Description |
|------|----------|-------------|
| `tokenizer.json` | `training/tokenizer/` | Trained BPE tokenizer (16K vocab) |

### Training Artifacts (HPC: ~/hpc-share/quantum-llm/)

| File | Location | Description | Size |
|------|----------|-------------|------|
| `final_model.pt` | `model/` | Final model weights | 5 MB |
| `best_model.pt` | `model/` | Best checkpoint (same as final) | 15 MB |
| `checkpoint_epoch1.pt` | `model/` | Epoch 1 checkpoint | 15 MB |
| `checkpoint_epoch2.pt` | `model/` | Epoch 2 checkpoint | 15 MB |
| `checkpoint_epoch3.pt` | `model/` | Epoch 3 checkpoint | 15 MB |
| `config.json` | `model/` | Model config | 149 B |
| `tokenizer.json` | `/` | Tokenizer (copied) | - |

### Scripts Created

| Script | Location | Purpose |
|--------|----------|---------|
| `train_tokenizer.py` | `training/scripts/` | Train BPE tokenizer |
| `inspect_data.py` | `training/scripts/` | Inspect combined Q&A and books |
| `inspect_stackexchange.py` | `training/scripts/` | Detailed Stack Exchange inspection |
| `inspect_book.py` | `training/scripts/` | Detailed book text inspection |
| `clean_stackexchange.py` | `training/scripts/` | Clean Stack Exchange data |
| `clean_book.py` | `training/scripts/` | Clean book texts |
| `clean_chatgpt_qa.py` | `training/scripts/` | Remove template prefixes |
| `clean_chatgpt_final.py` | `training/scripts/` | Remove duplicates and short entries |
| `combine_qa.py` | `training/scripts/` | Combine Q&A sources |
| `extract_books.py` | `training/scripts/` | Extract text from PDFs |
| `model.py` | HPC `scripts/` | Transformer architecture (RoPE, RMSNorm, SwiGLU) |
| `dataset.py` | HPC `scripts/` | Data loading for Q&A + books |
| `train.py` | HPC `scripts/` | Training loop with mixed precision |
| `train_job.sh` | HPC `/` | SLURM job script for dgxh partition |

---

## What Is Next

**Immediate next task:** QA evaluation of trained model

### Remaining Phase 1 Tasks

| Task | Priority | Status |
|------|----------|--------|
| Set up HPC environment (venv, PyTorch) | - | ✅ Done |
| Implement transformer architecture | - | ✅ Done |
| Write training script | - | ✅ Done |
| Train model on HPC | - | ✅ Done |
| Create QC test set (50-100 questions) | **Next** | ⬜ Pending |
| Evaluate (perplexity, test set) | After test set | ⬜ Pending |
| Document results for portfolio | After evaluation | ⬜ Pending |

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
| Load ChatGPT synthetic Q&A | ✅ Done |
| Extract text from book PDFs | ✅ Done |
| Preprocess and combine all sources | ✅ Done |
| Clean and verify all data | ✅ Done |
| Train custom tokenizer | ✅ Done |
| Set up HPC environment | ✅ Done |
| Implement transformer architecture | ✅ Done |
| Write training script | ✅ Done |
| Train model on HPC | ✅ Done |
| Create QC test set | ⬜ Pending |
| Evaluate model | ⬜ Pending |
| Document results | ⬜ Pending |

### Phase 2: RAG System

Build retrieval-augmented generation pipeline.

| Task | Status |
|------|--------|
| Process documents into chunks | ⬜ Pending |
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

2. **ChatGPT synthetic data had massive duplication.** Original 98K pairs had only ~2K unique answers (96% duplicates). After regeneration and cleanup: 85,643 pairs with 100% unique answers.

3. **PDF extraction produces garbage.** Raw text extraction creates fragments from math formulas, TOC dots, page numbers. Required paragraph-level filtering to remove fragments <50 chars.

4. **Stack Exchange data was mostly clean.** Only 11 rows needed removal (1 with HTML entities, 10 with very short answers).

5. **Chinchilla scaling matters.** Original 10M param target was too large for ~15M tokens. Reduced to 1.2M params for ~12.5:1 token-to-param ratio.

6. **Inspect after every step.** Early pipeline ran garbage through tokenizer. Proper workflow: acquire → inspect → clean → verify → proceed.

7. **Test locally before submitting HPC jobs.** Wasted 2 job submissions due to missing dependencies (tensorboard). Always verify imports and run a minimal test on submit node first.

8. **Use absolute paths in SLURM scripts.** Module loading and venv activation can fail silently. Use full paths to Python executable.

9. **H100s are fast.** 1.2M param model trained in 8 minutes at 626K tokens/sec. Could have used a larger model.

10. **Small models learn vocabulary, not reasoning.** Model outputs quantum jargon but answers are incoherent. RAG is essential for usability.

---

## Data Quality Verification Checklist

All data has been verified clean:

### Q&A Data (combined_qa_final.csv)
- [x] No null/empty questions or answers
- [x] No duplicate questions
- [x] No duplicate answers
- [x] No HTML entities
- [x] No very short entries (<20 char questions, <50 char answers)
- [x] Source distribution verified

### Book Data (combined_books.txt)
- [x] No very short paragraphs (<50 chars)
- [x] No control characters
- [x] No pure garbage lines (TOC dots, page numbers)
- [x] Each book inspected individually

### Tokenizer
- [x] All test sentences encode/decode correctly
- [x] Key QC terms are efficient single tokens
- [x] Special tokens assigned correct IDs

---

## Notes for Next Session

- Model is trained but outputs are incoherent (expected for 1.2M params)
- Need to create QC test set (50-100 questions) for proper evaluation
- Copy model files from HPC to local:
  ```powershell
  scp pereze4@submit-b.hpc.engr.oregonstate.edu:~/hpc-share/quantum-llm/model/final_model.pt E:\Personal_projects\Quantum-Computing-LLM\training\model\
  scp pereze4@submit-b.hpc.engr.oregonstate.edu:~/hpc-share/quantum-llm/model/config.json E:\Personal_projects\Quantum-Computing-LLM\training\model\
  ```
- Copy training scripts to local for version control
- Consider: Is the model good enough for portfolio, or retrain with different approach?

---

*Document version: 3.0*
