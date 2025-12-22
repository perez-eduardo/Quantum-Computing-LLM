# Quantum Computing Assistant - Implementation Plan

**Last Updated:** December 21, 2025

**Related Documents:**
- Design Document: `quantum-computing-assistant-design.md`
- Infrastructure Planning: `initial-exploratory-brainstorming.md`
- Model Investigation Report: `model_investigation_report.md`

---

## Current Status

**Phase:** 1 (Training Pipeline)
**Status:** Model evaluation complete. Data quality issues found. Cleaning before retraining.

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
| **Clean ChatGPT (remove boilerplate phrases)** | ✅ Done | `chatgpt_qa_cleaned_v2.csv` (85,583 pairs) |
| Combine Q&A sources | ✅ Done | `combined_qa_final.csv` (96,245 pairs after boilerplate clean) |
| Combine book texts | ✅ Done | `books/cleaned/combined_books.txt` |
| Train custom BPE tokenizer | ✅ Done | `tokenizer/tokenizer.json` (16K vocab) |
| **Filter truncated examples (>1024 tokens)** | ⬜ Pending | ~1,700 to remove |
| **Deduplicate templated Q&A** | ⬜ Pending | ~20,000+ to remove |

### HPC Training

| Task | Status | Notes |
|------|--------|-------|
| Set up HPC environment | ✅ Done | Python 3.11 venv, PyTorch 2.5.1+cu121 |
| Implement transformer architecture | ✅ Done | `scripts/model.py` |
| Write dataset loader | ✅ Done | `scripts/dataset.py` |
| Write training script | ✅ Done | `scripts/train.py` |
| Create SLURM job script | ✅ Done | `train_job.sh` |
| Train model on H100 (initial) | ✅ Done | 4 minutes, 3 epochs |
| **Evaluate model quality** | ✅ Done | 14.8% keyword match, incoherent outputs |
| **Investigate data quality issues** | ✅ Done | Found boilerplate, templates, truncation issues |
| **Retrain with cleaned data** | ⬜ Pending | 10 epochs planned |

### Model Investigation

| Task | Status | Findings |
|------|--------|----------|
| Evaluate with 50-question test set | ✅ Done | 14.8% keyword match |
| Compare checkpoints (epoch 1, 3, 6) | ✅ Done | Epoch 6 >> 3 >> 1, more training helps |
| Analyze token length distribution | ✅ Done | 88% under 128 tokens, 5.5% truncated |
| Inspect boilerplate in ChatGPT data | ✅ Done | 83.4% contained boilerplate phrases |
| Clean boilerplate from ChatGPT | ✅ Done | 0% boilerplate remaining |
| Inspect truncated examples | ✅ Done | 100% from Stack Exchange, code dumps |
| Inspect short examples | ✅ Done | 30% are templated repetitive garbage |
| Analyze book data contribution | ✅ Done | Only 3.8% of training examples |

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

## Training Results (Initial Run)

**Job:** 19739587 on dgxh-1 (H100 80GB)
**Duration:** ~4 minutes
**Throughput:** ~626K tokens/sec

| Epoch | Train Loss | Val Loss | Perplexity |
|-------|------------|----------|------------|
| 1 | 5.32 | 3.23 | 25.29 |
| 2 | 2.94 | 2.82 | 16.84 |
| 3 | 2.74 | 2.74 | 15.55 |

**Training Config:**
| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size | 64 |
| Max LR | 3e-4 |
| Min LR | 3e-5 |
| Warmup ratio | 0.1 |
| Max seq length | 512 |

**Issues Identified:**
- Loss still decreasing at epoch 3 (undertrained)
- LR decayed to minimum before convergence
- Only 4,461 steps total

---

## Model Evaluation Results

**Evaluation Score:** 14.8% keyword match (50 questions)

**By Category:**
| Category | Score |
|----------|-------|
| basics | 32.6% |
| gates | 20.8% |
| algorithms | 18.0% |
| entanglement | 9.0% |
| superposition | 9.0% |
| hardware | 8.0% |
| applications | 6.7% |
| measurement | 4.2% |

**By Difficulty:**
| Difficulty | Score |
|------------|-------|
| easy | 20.2% |
| medium | 12.4% |
| hard | 11.9% |

**Qualitative Observations:**
- Outputs contain quantum terminology
- Responses are incoherent fragments stitched together
- Model rambles, doesn't answer the actual question
- Broken LaTeX, trails off mid-sentence
- Heavy boilerplate appended to every answer

---

## Checkpoint Comparison

Tested epochs 1, 3, and 6 (from a cancelled 6-epoch job).

| Epoch | Quality | Coherence | Boilerplate |
|-------|---------|-----------|-------------|
| 1 | Worst | Fragments, broken LaTeX | Present |
| 3 | Medium | Some complete sentences | Heavy |
| 6 | Best | Good sentences, some correct answers | Heavy |

**Key Finding:** Model improves significantly with more epochs. Epoch 6 outputs are substantially better than epoch 3.

**Example (Epoch 6):**
```
Q: What is a CNOT gate?
A: CNOT flips the target qubit if the control is |1⟩; it is widely used to create entanglement and implement parity checks.
```

---

## Data Quality Issues Found

### Issue 1: Boilerplate in ChatGPT Data (✅ FIXED)

- **83.4%** of ChatGPT answers contained boilerplate phrases
- **65.9%** had 2+ boilerplate phrases appended
- Common patterns:
  - "In fault-tolerant settings..."
  - "In practical workflows..."
  - "In NISQ experiments..."
  - "In many tutorials..."
  - "In simulators..."
  - "On real devices..."

**Action Taken:** Ran `clean_boilerplate.py`, reduced to 0% boilerplate.

### Issue 2: Truncated Examples (5.5%)

- **5,290 examples** exceed 512 token limit
- **100% from Stack Exchange**
- Content is garbage: code dumps, installation logs, JSON outputs
- Model sees question + partial answer, never the conclusion

| Tokens Over Limit | Count | Percent |
|-------------------|-------|---------|
| 1-100 | 1,120 | 21.2% |
| 101-500 | 2,489 | 47.1% |
| 501-1000 | 1,103 | 20.9% |
| 1001-5000 | 569 | 10.8% |
| 5000+ | 9 | 0.2% |

**Recommendation:** Filter out examples >1024 tokens.

### Issue 3: Templated Repetitive Examples (30%)

- **29.9% very short** (<50 tokens): 28,823 examples
- **100% from ChatGPT synthetic**
- Same templates repeated with different numbers:

```
Q: Why does the state space grow exponentially with 198 qubits?
A: Because each additional qubit doubles the dimension... 2^198 basis states

Q: Why does the state space grow exponentially with 319 qubits?
A: Because each additional qubit doubles the dimension... 2^319 basis states
```

Model memorizes templates, not concepts.

**Recommendation:** Deduplicate, keep 1-2 examples per template pattern.

### Issue 4: Book Data Drowned Out

- **3,830 book chunks** vs 96,245 Q&A pairs
- Book data is only **3.8%** of training examples
- 25:1 ratio (Q&A to book)

**Recommendation:** Will naturally improve after filtering Q&A templates.

---

## Data Cleaning Status

| Issue | Severity | Status | Action |
|-------|----------|--------|--------|
| Boilerplate in 83% of ChatGPT | High | ✅ Fixed | Cleaned, 0% remaining |
| Truncated examples (5.5%) | High | ⬜ Pending | Filter >1024 tokens |
| Templated repetitive (30%) | High | ⬜ Pending | Deduplicate patterns |
| Book data only 3.8% | Medium | ⬜ Will fix | Improves after Q&A filtering |
| Undertrained (3 epochs) | Medium | ⬜ Pending | Increase to 10 epochs |

---

## Current Data Summary

### After Boilerplate Cleaning

| Source | Rows |
|--------|------|
| ChatGPT (cleaned) | 85,583 |
| Stack Exchange | 10,662 |
| **Total Q&A** | **96,245** |
| Book chunks | 3,830 |

### Expected After All Cleaning

| Metric | Before | After (Est.) |
|--------|--------|--------------|
| Total Q&A rows | 96,245 | ~70,000 |
| Templated examples | ~30% | <5% |
| Truncated garbage | 5.5% | 0% |
| Book data ratio | 3.8% | ~5.5% |
| Boilerplate | 0% | 0% |

---

## Retraining Plan

### Hyperparameter Changes

| Parameter | Old | New | Reason |
|-----------|-----|-----|--------|
| Epochs | 3 | 10 | Loss still dropping, checkpoint 6 much better |
| Min LR | 3e-5 | 1e-5 | Allow finer convergence |
| Warmup ratio | 0.1 | 0.05 | Shorter warmup, more training at peak LR |

### Data Changes

1. ✅ Remove boilerplate (done)
2. ⬜ Filter examples >1024 tokens
3. ⬜ Deduplicate templated examples

### Job Script

```bash
# train_job_v2.sh
#SBATCH --job-name=qllm-v2
#SBATCH --partition=dgxh
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/train_%j.out

~/hpc-share/quantum-llm/venv/bin/python scripts/train.py \
    --epochs 10 \
    --min_lr 1e-5 \
    --warmup_ratio 0.05
```

---

## What Is Next

**Immediate next task:** Filter truncated examples, deduplicate templates, retrain

### Remaining Phase 1 Tasks

| Task | Priority | Status |
|------|----------|--------|
| Set up HPC environment | - | ✅ Done |
| Implement transformer architecture | - | ✅ Done |
| Write training script | - | ✅ Done |
| Train model on HPC (initial) | - | ✅ Done |
| Evaluate model | - | ✅ Done |
| Investigate data quality | - | ✅ Done |
| Clean boilerplate from ChatGPT | - | ✅ Done |
| Filter truncated examples (>1024 tokens) | **Next** | ⬜ Pending |
| Deduplicate templated Q&A | **Next** | ⬜ Pending |
| Retrain with cleaned data (10 epochs) | After cleaning | ⬜ Pending |
| Re-evaluate model | After retraining | ⬜ Pending |
| Document results for portfolio | After evaluation | ⬜ Pending |

---

## Output Files Created

### Data Files

| File | Location | Description | Size |
|------|----------|-------------|------|
| `combined_qa_final.csv` | HPC `data/` | All Q&A pairs (boilerplate cleaned) | ~54 MB |
| `chatgpt_qa_cleaned_v2.csv` | `data/raw/` | ChatGPT Q&A (boilerplate cleaned) | ~36 MB |
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
| `tokenizer.json` | `/` | Tokenizer (copied) | - |

### Investigation Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| `sanity_check.py` | HPC `scripts/` | Quick model output test |
| `evaluate_model.py` | HPC `scripts/` | Full evaluation with test set |
| `analyze_token_lengths.py` | HPC `scripts/` | Token distribution analysis |
| `inspect_boilerplate.py` | `training/scripts/` | Find boilerplate patterns |
| `clean_boilerplate.py` | `training/scripts/` | Strip boilerplate from answers |
| `inspect_truncated.py` | HPC `scripts/` | Examine truncated examples |
| `inspect_short.py` | HPC `scripts/` | Examine short/templated examples |
| `inspect_books.py` | HPC `scripts/` | Analyze book data contribution |

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
| Evaluate model | ✅ Done |
| **Investigate data quality issues** | ✅ Done |
| **Clean data (boilerplate, truncation, templates)** | 🔄 In Progress |
| **Retrain model** | ⬜ Pending |
| **Re-evaluate model** | ⬜ Pending |
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

9. **H100s are fast.** 1.2M param model trained in 4 minutes at 626K tokens/sec. Could have used a larger model.

10. **Small models learn vocabulary, not reasoning.** Model outputs quantum jargon but answers are incoherent. RAG is essential for usability.

11. **ChatGPT synthetic data had hidden boilerplate.** 83% of answers contained repetitive boilerplate phrases that the model memorized. Always inspect generated data thoroughly.

12. **Templated data teaches templates.** 30% of ChatGPT data was the same question with different numbers. Model memorizes patterns instead of learning concepts.

13. **Truncated examples are worse than no examples.** 5.5% of data exceeded token limit. These become partial answers with no conclusion, confusing the model.

14. **More epochs help small models.** Checkpoint comparison showed epoch 6 >> epoch 3 >> epoch 1. Initial 3 epochs was undertrained.

15. **Garbage in, garbage out.** Multiple rounds of data cleaning required. Initial "clean" data still had major issues only visible after model evaluation.

---

## Data Quality Verification Checklist

### Q&A Data (combined_qa_final.csv)
- [x] No null/empty questions or answers
- [x] No duplicate questions
- [x] No duplicate answers
- [x] No HTML entities
- [x] No very short entries (<20 char questions, <50 char answers)
- [x] Source distribution verified
- [x] ~~No boilerplate phrases~~ ✅ Fixed
- [ ] No truncated garbage (>1024 tokens)
- [ ] No templated repetitive examples

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

- Boilerplate cleaning complete, uploaded to HPC as `combined_qa_final.csv`
- Still need to filter truncated examples (>1024 tokens)
- Still need to deduplicate templated Q&A examples
- After cleaning, retrain with 10 epochs, min_lr 1e-5, warmup 0.05
- Checkpoints from cancelled 6-epoch job available for comparison
- Investigation report saved as `model_investigation_report.md`

---

*Document version: 4.0*
