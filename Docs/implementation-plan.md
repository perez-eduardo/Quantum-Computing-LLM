# Quantum Computing Assistant - Implementation Plan

**Last Updated:** December 24, 2025

**Related Documents:**
- Design Document: `quantum-computing-assistant-design.md`
- Infrastructure Planning: `initial-exploratory-brainstorming.md`
- Model Investigation Report: `model_investigation_report.md`

---

## Current Status

**Phase 1:** Training Pipeline - ✅ COMPLETE (v5 trained)
**Phase 2:** RAG System - ✅ COMPLETE
**Phase 3:** Backend - ⬜ IN PROGRESS

**Next Action:** Connect RAG to model for end-to-end testing

---

## Model v5 Training Complete (December 24, 2025)

### Two-Phase Training Results

**Phase 1: Book Pretraining**
| Metric | Value |
|--------|-------|
| Data | combined_books_cleaned.txt (620K words) |
| Epochs | 17 |
| Final Perplexity | 2.20 |
| Time | ~13 min on H100 |

**Phase 2: Context Q&A Fine-tuning**
| Metric | Value |
|--------|-------|
| Data | 28,071 context-format Q&A pairs |
| Epochs | 10 |
| Time | ~116 min on H100 |

### Evaluation Results

| Test Type | Score |
|-----------|-------|
| With context (RAG simulation) | **64% keyword accuracy** |
| Without context | Gibberish (expected) |

**Verdict:** ✅ Model successfully learned to use context

### Sample Outputs (With Context)

| Question | Keywords Found | Score |
|----------|----------------|-------|
| What is a qubit? | qubit, quantum, bit, superposition | 80% |
| Why is quantum computing important for cryptography? | cryptography, encryption, security | 60% |
| What is a quantum circuit? | circuit, gate, qubit, quantum | 80% |
| Why is quantum error correction needed? | error, correct, fault | 60% |

---

## What Has Been Done

### Data Acquisition

| Task | Status | Notes |
|------|--------|-------|
| Download Stack Exchange QC dump | ✅ Done | 28K total posts |
| Obtain book PDFs | ✅ Done | 5 books |
| Generate Claude Q&A | ✅ Done | 15,000 pairs across 38 batches |
| Obtain CoT Reasoning Dataset | ✅ Done | 2,998 Q&A pairs with chain-of-thought |
| Clean books | ✅ Done | 620K words (removed copyright, TOC, spam) |

### Data Processing & Cleaning

| Task | Status | Output |
|------|--------|--------|
| Process Stack Exchange XML | ✅ Done | 10,673 pairs |
| Extract and clean book texts | ✅ Done | 620,455 words |
| Generate Claude Q&A | ✅ Done | 15,000 pairs |
| Process CoT dataset | ✅ Done | 2,998 pairs |
| Train custom BPE tokenizer | ✅ Done | 16K vocab |
| Add context to CoT dataset | ✅ Done | Reasoning from metadata |
| Add context to Stack Exchange | ✅ Done | Tags + body as context |
| Add context to Claude Q&A | ✅ Done | Topic-matched relevant Q&A pairs |

### Context-Format Data

| Source | Rows | Context Type |
|--------|------|--------------|
| cot_qa_context.csv | 2,998 | Chain-of-thought reasoning |
| stackexchange_qa_context.csv | 10,673 | Tags + question body |
| claude_qa_context.csv | 14,400 | Topic-matched relevant Q&A pairs |
| **Total** | **28,071** | |

### HPC Training

| Task | Status | Notes |
|------|--------|-------|
| Set up HPC environment | ✅ Done | Python 3.11 venv, PyTorch 2.5.1+cu121 |
| Implement transformer architecture | ✅ Done | 125.8M params |
| Train tokenizer | ✅ Done | 16K vocab BPE |
| Phase 1: Book pretraining | ✅ Done | 17 epochs, perplexity 2.20 |
| Phase 2: Context Q&A fine-tuning | ✅ Done | 10 epochs |
| Evaluate model | ✅ Done | 64% keyword accuracy with context |
| Download model files | ⬜ Pending | |

### RAG System

| Task | Status | Notes |
|------|--------|-------|
| Set up Neon database with pgvector | ✅ Done | Free tier, 512MB limit |
| Embed Q&A pairs | ✅ Done | 26,764 pairs embedded |
| Test retrieval | ✅ Done | 94% pass rate |

---

## Training Results Comparison

| Metric | v1 | v3 | v4 | v5 |
|--------|----|----|----|----|
| Parameters | 1.2M | 1.2M | 1.2M | **125.8M** |
| Training data | 96K garbage | 24K plain | 26K plain | 28K context |
| Format | Plain Q&A | Plain Q&A | Plain Q&A | **Context-aware** |
| Perplexity | 15.55 | 89.63 | 91.80 | **2.20 (Phase 1)** |
| Keyword Accuracy | 14.8% | 16.4% | 11.4% | **64%** |
| Can use RAG context | ❌ | ❌ | ❌ | **✅** |

---

## What Is Next

### Immediate: End-to-End Integration

| Task | Priority | Status |
|------|----------|--------|
| Download model from HPC | High | ⬜ Pending |
| Create inference pipeline | High | ⬜ Pending |
| Connect RAG retrieval to model | High | ⬜ Pending |
| Test end-to-end flow | High | ⬜ Pending |

### Phase 3: Backend

| Task | Priority | Status |
|------|----------|--------|
| Implement RAG + model pipeline | High | ⬜ Pending |
| Create FastAPI endpoints | High | ⬜ Pending |
| Test end-to-end flow | High | ⬜ Pending |

### Phase 4: Frontend

| Task | Priority | Status |
|------|----------|--------|
| Single HTML page | Medium | ⬜ Pending |
| API integration | Medium | ⬜ Pending |

### Phase 5: Deployment

| Task | Priority | Status |
|------|----------|--------|
| Deploy to Railway | Medium | ⬜ Pending |
| Set spending caps | Medium | ⬜ Pending |

---

## Development Phases

### Phase 1: Training Pipeline ✅ COMPLETE

| Task | Status |
|------|--------|
| Generate Claude Q&A (15,000 pairs) | ✅ Done |
| Process Stack Exchange | ✅ Done |
| Process CoT dataset (2,998 pairs) | ✅ Done |
| Add context to all datasets | ✅ Done |
| Clean books (620K words) | ✅ Done |
| Train tokenizer | ✅ Done |
| Phase 1: Book pretraining (17 epochs) | ✅ Done |
| Phase 2: Context Q&A fine-tuning (10 epochs) | ✅ Done |
| Evaluate model (64% accuracy) | ✅ Done |

### Phase 2: RAG System ✅ COMPLETE

| Task | Status |
|------|--------|
| Set up Neon database with pgvector | ✅ Done |
| Embed Q&A pairs (26,764) | ✅ Done |
| Test retrieval quality | ✅ Done |
| Achieve 94% pass rate | ✅ Done |

### Phase 3: Backend ⬜ IN PROGRESS

| Task | Status |
|------|--------|
| Download model from HPC | ⬜ Pending |
| FastAPI endpoints | ⬜ Pending |
| Model inference | ⬜ Pending |
| RAG integration | ⬜ Pending |
| End-to-end testing | ⬜ Pending |

---

## Output Files

### Model Files (HPC)

| File | Location | Description |
|------|----------|-------------|
| `phase1_best.pt` | `~/hpc-share/quantum-llm/model/` | Book pretrained |
| `phase2_best.pt` | `~/hpc-share/quantum-llm/model/` | Context fine-tuned |
| `final_model.pt` | `~/hpc-share/quantum-llm/model/` | **Production model** |
| `tokenizer.json` | `~/hpc-share/quantum-llm/` | BPE tokenizer |

### Training Scripts (HPC)

| File | Purpose |
|------|---------|
| `scripts/model.py` | 125.8M param transformer architecture |
| `scripts/dataset.py` | Book + Context Q&A data loading |
| `scripts/train.py` | Two-phase training logic |
| `scripts/train_phase1.sh` | SLURM job for book pretraining |
| `scripts/train_phase2.sh` | SLURM job for context fine-tuning |
| `scripts/evaluate.py` | Model evaluation |
| `scripts/train_tokenizer.py` | BPE tokenizer training |

### Context-Format Training Data

| File | Rows | Description |
|------|------|-------------|
| `cot_qa_context.csv` | 2,998 | question, answer, context (reasoning) |
| `stackexchange_qa_context.csv` | 10,673 | question, answer, context (tags + body) |
| `claude_qa_context.csv` | 14,400 | question, answer, context (topic-matched) |

### Database (Neon)

| Table | Contents |
|-------|----------|
| chunks | 26,764 Q&A embeddings |

---

## Key Findings During Implementation

1. **ChatGPT synthetic data was 94% garbage.** Abandoned entirely.

2. **Claude Q&A generation works.** 15,000 pairs with proper verification.

3. **Small models cannot reason.** 1.2M params produces gibberish. 125.8M produces coherent text.

4. **Two-phase training works.** Book pretraining (perplexity 2.20) + context fine-tuning = usable model.

5. **Context format is critical.** Model must be trained on same format used at inference.

6. **Topic-matched context beats random.** Context must be relevant to the question.

7. **64% keyword accuracy with context.** Major improvement from v4's 11.4%.

8. **Model needs context to function.** Without context, outputs are gibberish. RAG is essential.

9. **Q&A pairs beat book chunks for RAG.** 94% retrieval accuracy.

---

*Document version: 13.0*
