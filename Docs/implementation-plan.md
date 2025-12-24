# Quantum Computing Assistant - Implementation Plan

**Last Updated:** December 24, 2025

**Related Documents:**
- Design Document: `quantum-computing-assistant-design.md`
- Infrastructure Planning: `initial-exploratory-brainstorming.md`
- Model Investigation Report: `model_investigation_report.md`

---

## Current Status

**Phase 1:** Training Pipeline - ✅ COMPLETE
**Phase 2:** RAG System - ✅ COMPLETE
**Phase 3:** Backend - ⬜ IN PROGRESS

**Next Action:** Connect RAG retrieval to model for end-to-end testing

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
| Obtain CoT Reasoning Dataset | ✅ Done | 2,756 Q&A pairs with chain-of-thought |

### Data Processing & Cleaning

| Task | Status | Output |
|------|--------|--------|
| Process Stack Exchange XML | ✅ Done | 10,673 pairs |
| Filter Stack Exchange (>1024 tokens) | ✅ Done | 9,008 pairs |
| Extract and clean book texts | ✅ Done | 633,562 words |
| Generate Claude Q&A | ✅ Done | 15,000 pairs |
| Process CoT dataset | ✅ Done | 2,756 pairs |
| Combine all sources | ✅ Done | 26,764 Q&A pairs |
| Train custom BPE tokenizer | ✅ Done | 16K vocab |

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

### RAG System

| Task | Status | Notes |
|------|--------|-------|
| Set up Neon database with pgvector | ✅ Done | Free tier, 512MB limit |
| Embed book chunks | ✅ Done | 2,847 chunks (later removed) |
| Test initial retrieval | ✅ Done | 92% pass rate |
| Embed Q&A pairs | ✅ Done | 26,764 pairs embedded |
| Test improved retrieval | ✅ Done | 94% pass rate |
| Remove book chunks | ✅ Done | Q&A pairs are better quality |
| Attempt hybrid search | ❌ Skipped | Storage limit hit, 94% is sufficient |

---

## Training Results

### v1 (Garbage Data - December 21, 2025)

| Metric | Value |
|--------|-------|
| Data | 96K Q&A (94% ChatGPT garbage) |
| Epochs | 3 |
| Perplexity | 15.55 |
| Eval Score | 14.8% keyword match |
| Boilerplate | 83.4% contaminated |

### v3 (Clean Data - December 23, 2025)

| Metric | Value |
|--------|-------|
| Data | 24K Q&A (Claude + Stack Exchange) |
| Epochs | 10 |
| Perplexity | 89.63 |
| Eval Score | 16.4% keyword match |
| Boilerplate | 0% contaminated |

### v4 (Expanded Data - December 24, 2025)

| Metric | Value |
|--------|-------|
| Data | 26,764 Q&A (Claude + Stack Exchange + CoT) |
| Epochs | 10 |
| Perplexity | 91.80 |
| Eval Score | 11.4% keyword match |
| Boilerplate | 0% contaminated |

---

## RAG Results

### Retrieval Quality

| Version | Pass Rate | Notes |
|---------|-----------|-------|
| v1 (books only) | 92% | Book chunks mention terms without defining |
| v2 (books + Q&A) | 94% | Q&A pairs have actual definitions |
| v2 (Q&A only) | 94% | Removed book chunks, same quality |

### Database Contents

| Content | Count | Status |
|---------|-------|--------|
| Book chunks | 0 | Removed (Q&A pairs are better) |
| Q&A pairs | 26,764 | Active |
| **Total** | **26,764** | |

### Q&A Sources in RAG

| Source | Count |
|--------|-------|
| claude_synthetic | 15,000 |
| stackexchange | 9,008 |
| cot_reasoning | 2,756 |

### Remaining Failures (6/100)

| Query | Issue |
|-------|-------|
| Computational basis | Retrieved question, not definition |
| Fredkin gate | Data gap |
| QAOA | Acronym confusion with QBism |
| Quantum counting | Retrieved excerpt, not definition |
| Partial trace | Retrieved related but not definition |
| Fidelity | Retrieved "layer fidelity" variant |

---

## What Is Next

**Immediate next task:** End-to-end testing (connect RAG to model)

### Phase 3: Backend

| Task | Priority | Status |
|------|----------|--------|
| Determine inference approach | High | ⬜ Pending |
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
| Process CoT dataset (2,756 pairs) | ✅ Done |
| Combine all sources | ✅ Done |
| Train model v4 | ✅ Done |
| Evaluate model v4 | ✅ Done |
| Download model files | ✅ Done |

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
| FastAPI endpoints | ⬜ Pending |
| Model inference | ⬜ Pending |
| RAG integration | ⬜ Pending |
| End-to-end testing | ⬜ Pending |

### Phase 4: Frontend

| Task | Status |
|------|--------|
| Single HTML page | ⬜ Pending |
| API integration | ⬜ Pending |

### Phase 5: Deployment

| Task | Status |
|------|--------|
| Deploy to Railway | ⬜ Pending |
| Set spending caps | ⬜ Pending |

---

## Output Files

### Local (training/model/)

| File | Description |
|------|-------------|
| `final_model.pt` | v4 model weights |
| `best_model.pt` | Best model by val loss |
| `config.json` | Model config |
| `tokenizer.json` | BPE tokenizer (16K vocab) |

### Database (Neon)

| Table | Contents |
|-------|----------|
| chunks | 26,764 Q&A embeddings |

---

## Key Findings During Implementation

1. **ChatGPT synthetic data was 94% garbage.** Boilerplate in 83%, templates in 59%. Abandoned entirely.

2. **Claude Q&A generation works.** 15,000 pairs with 100% unique questions, proper verification.

3. **More epochs help.** Checkpoint comparison showed clear progression epoch 1 → 5 → 10.

4. **Higher perplexity can be better.** v1's low perplexity came from memorizing garbage.

5. **Small models learn vocabulary, not reasoning.** 1.2M params produces quantum jargon but incoherent answers. RAG essential.

6. **Data quality verification is critical.** Boilerplate check confirmed 0% contamination.

7. **H100s are fast.** 10 epochs in 13 minutes at 620K tokens/sec.

8. **Q&A pairs beat book chunks for RAG.** Book chunks mention terms without defining them. Q&A pairs have actual definitions.

9. **94% retrieval is achievable.** Remaining failures are data gaps and semantic edge cases.

---

*Document version: 11.0*
