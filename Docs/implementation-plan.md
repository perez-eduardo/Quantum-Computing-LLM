# Quantum Computing Assistant - Implementation Plan

**Last Updated:** December 24, 2025

**Related Documents:**
- Design Document: `quantum-computing-assistant-design.md`
- Infrastructure Planning: `initial-exploratory-brainstorming.md`
- Model Investigation Report: `model_investigation_report.md`

---

## Current Status

**Phase 1:** Training Pipeline - ⚠️ REQUIRES RETRAINING
**Phase 2:** RAG System - ✅ COMPLETE
**Phase 3:** Backend - ⬜ BLOCKED (awaiting model retraining)

**Next Action:** Retrain model with context-aware format (50M-100M params)

---

## CRITICAL DESIGN FLAW DISCOVERED (December 24, 2025)

### The Problem

The 1.2M parameter model was trained on plain Q&A format:
```
Q: What is superposition?
A: Superposition allows...
```

But at inference time, RAG retrieves context that the model cannot use:
```
Context: [retrieved Q&A pairs]
Question: What is superposition?
```

**Result:** The model ignores context entirely because it never learned the format.

### The Solution

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

2. **Scale model to 50M-100M parameters** for coherent generation

3. **Deploy on Railway** (still fits, ~100-200MB for 50M-100M params)

---

## What Has Been Done

### Data Acquisition

| Task | Status | Notes |
|------|--------|-------|
| Download QuantumLLMInstruct | ✅ Done | Only 46 usable pairs. Excluded. |
| Download Stack Exchange QC dump | ✅ Done | 28K total posts |
| ~~Obtain ChatGPT synthetic Q&A~~ | ❌ Abandoned | 94% garbage. Replaced with Claude Q&A. |
| Obtain book PDFs | ✅ Done | 5 books |
| Generate Claude Q&A | ✅ Done | 15,000 pairs across 38 batches |
| Obtain CoT Reasoning Dataset | ✅ Done | 2,998 Q&A pairs with chain-of-thought |

### Data Processing & Cleaning

| Task | Status | Output |
|------|--------|--------|
| Process Stack Exchange XML | ✅ Done | 10,673 pairs |
| Filter Stack Exchange (>1024 tokens) | ✅ Done | 9,008 pairs |
| Extract and clean book texts | ✅ Done | 633,562 words |
| Generate Claude Q&A | ✅ Done | 15,000 pairs |
| Process CoT dataset | ✅ Done | 2,998 pairs |
| Train custom BPE tokenizer | ✅ Done | 16K vocab |
| **Add context to CoT dataset** | ✅ Done | Reasoning from metadata |
| **Add context to Stack Exchange** | ✅ Done | Tags + body as context |
| **Add context to Claude Q&A** | ✅ Done | Template-based context (38 batches) |

### Context-Format Data (NEW)

| Source | Rows | Context Type |
|--------|------|--------------|
| cot_qa_context.csv | 2,998 | Chain-of-thought reasoning |
| stackexchange_qa_context.csv | 10,673 | Tags + question body |
| claude_qa_batch1-38_context.csv | 15,000 | Template-based (question type + topics) |
| **Total** | **28,671** | |

### HPC Training

| Task | Status | Notes |
|------|--------|-------|
| Set up HPC environment | ✅ Done | Python 3.11 venv, PyTorch 2.5.1+cu121 |
| Implement transformer architecture | ✅ Done | `scripts/model.py` |
| Train model v1 (garbage data) | ✅ Done | 3 epochs, perplexity 15.55, 14.8% eval |
| Investigate data quality issues | ✅ Done | ChatGPT data 94% garbage |
| Train model v3 (clean data) | ✅ Done | 10 epochs, perplexity 89.63, 16.4% eval |
| Train model v4 (expanded data) | ✅ Done | 10 epochs, perplexity 91.80, 11.4% eval |
| Download model files | ✅ Done | `training/model/` |
| **Update model.py for 50M-100M params** | ⬜ Pending | Required for coherent generation |
| **Retrain with context-aware format** | ⬜ Pending | New training format |

### RAG System

| Task | Status | Notes |
|------|--------|-------|
| Set up Neon database with pgvector | ✅ Done | Free tier, 512MB limit |
| Embed Q&A pairs | ✅ Done | 26,764 pairs embedded |
| Test retrieval | ✅ Done | 94% pass rate |

---

## Training Results (v1-v4, Pre-Redesign)

### v4 (Final before redesign - December 24, 2025)

| Metric | Value |
|--------|-------|
| Data | 26,764 Q&A (plain format, NO context) |
| Epochs | 10 |
| Perplexity | 91.80 |
| Eval Score | 11.4% keyword match |
| Boilerplate | 0% contaminated |

**Problem:** Cannot use RAG context at inference. Requires retraining.

---

## What Is Next

### Immediate: Model Retraining

| Task | Priority | Status |
|------|----------|--------|
| Generate context-format training data | High | ✅ Done |
| Update model.py to 50M-100M params | High | ⬜ Pending |
| Retrain on HPC (30-60 min) | High | ⬜ Pending |
| Test coherent generation | High | ⬜ Pending |
| Evaluate with context | High | ⬜ Pending |

### Phase 3: Backend (After Retraining)

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

### Phase 1: Training Pipeline ⚠️ REQUIRES RETRAINING

| Task | Status |
|------|--------|
| Generate Claude Q&A (15,000 pairs) | ✅ Done |
| Process Stack Exchange | ✅ Done |
| Process CoT dataset (2,998 pairs) | ✅ Done |
| **Add context to all datasets** | ✅ Done |
| Train model v4 (plain format) | ✅ Done |
| **Update architecture (50M-100M)** | ⬜ Pending |
| **Retrain with context format** | ⬜ Pending |

### Phase 2: RAG System ✅ COMPLETE

| Task | Status |
|------|--------|
| Set up Neon database with pgvector | ✅ Done |
| Embed Q&A pairs (26,764) | ✅ Done |
| Test retrieval quality | ✅ Done |
| Achieve 94% pass rate | ✅ Done |

### Phase 3: Backend ⬜ BLOCKED

| Task | Status |
|------|--------|
| FastAPI endpoints | ⬜ Pending |
| Model inference | ⬜ Pending |
| RAG integration | ⬜ Pending |
| End-to-end testing | ⬜ Pending |

---

## Output Files

### Context-Format Training Data (NEW)

| File | Rows | Description |
|------|------|-------------|
| `cot_qa_context.csv` | 2,998 | question, answer, context (reasoning) |
| `stackexchange_qa_context.csv` | 10,673 | question, answer, context (tags + body) |
| `claude_qa_batch1-38_context.csv` | 15,000 | question, answer, context (template) |

### Local (training/model/) - OUTDATED

| File | Description |
|------|-------------|
| `final_model.pt` | v4 model weights (plain format, cannot use context) |
| `best_model.pt` | Best model by val loss |
| `config.json` | Model config (needs update for 50M-100M) |
| `tokenizer.json` | BPE tokenizer (16K vocab) |

### Database (Neon)

| Table | Contents |
|-------|----------|
| chunks | 26,764 Q&A embeddings |

---

## Key Findings During Implementation

1. **ChatGPT synthetic data was 94% garbage.** Boilerplate in 83%, templates in 59%. Abandoned entirely.

2. **Claude Q&A generation works.** 15,000 pairs with 100% unique questions, proper verification.

3. **Small models learn vocabulary, not reasoning.** 1.2M params produces quantum jargon but incoherent answers.

4. **Data quality verification is critical.** Boilerplate check confirmed 0% contamination.

5. **Q&A pairs beat book chunks for RAG.** Book chunks mention terms without defining them.

6. **94% retrieval is achievable.** Remaining failures are data gaps and semantic edge cases.

7. **CRITICAL: Training format must match inference format.** Model trained on plain Q&A cannot use RAG context. Must retrain with context-aware format.

8. **Coherent generation requires capacity.** 50M-100M parameters needed for readable answers, not 1.2M.

---

*Document version: 12.0*
