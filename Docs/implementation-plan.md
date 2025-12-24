# Quantum Computing Assistant - Implementation Plan

**Last Updated:** December 24, 2025

**Related Documents:**
- Design Document: `quantum-computing-assistant-design.md`
- Infrastructure Planning: `initial-exploratory-brainstorming.md`
- Model Investigation Report: `model_investigation_report.md`

---

## Current Status

**Phase:** 1 (Training Pipeline) - ✅ COMPLETE
**Status:** Model v3 trained, evaluated, and verified. Ready for Phase 2.

---

## What Has Been Done

### Data Acquisition

| Task | Status | Notes |
|------|--------|-------|
| Download QuantumLLMInstruct | ✅ Done | Only 46 usable pairs. Excluded. |
| Download Stack Exchange QC dump | ✅ Done | 28K total posts |
| ~~Obtain ChatGPT synthetic Q&A~~ | ❌ Abandoned | 94% garbage. Replaced with Claude Q&A. |
| Obtain book PDFs | ✅ Done | 5 books |
| **Generate Claude Q&A** | ✅ Done | **15,000 pairs across 38 batches** |

### Data Processing & Cleaning

| Task | Status | Output |
|------|--------|--------|
| Process Stack Exchange XML | ✅ Done | 10,673 pairs |
| Filter Stack Exchange (>1024 tokens) | ✅ Done | 9,019 pairs |
| Extract and clean book texts | ✅ Done | 633,562 words |
| **Generate Claude Q&A** | ✅ Done | 15,000 pairs |
| Combine Q&A sources | ✅ Done | 24,019 pairs |
| Train custom BPE tokenizer | ✅ Done | 16K vocab |

### HPC Training

| Task | Status | Notes |
|------|--------|-------|
| Set up HPC environment | ✅ Done | Python 3.11 venv, PyTorch 2.5.1+cu121 |
| Implement transformer architecture | ✅ Done | `scripts/model.py` |
| Train model v1 (garbage data) | ✅ Done | 3 epochs, perplexity 15.55, 14.8% eval |
| Investigate data quality issues | ✅ Done | ChatGPT data 94% garbage |
| **Train model v3 (clean data)** | ✅ Done | **10 epochs, perplexity 89.63** |
| **Evaluate model v3** | ✅ Done | **16.4% keyword match** |
| **Verify data quality** | ✅ Done | **0% boilerplate contamination** |

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
| Eval Score | **16.4% keyword match** |
| Boilerplate | **0% contaminated** |

### v3 Loss Progression

| Epoch | Train Loss | Val Loss | Perplexity |
|-------|------------|----------|------------|
| 1 | 7.95 | 6.39 | 594.77 |
| 5 | 4.81 | 4.69 | 108.64 |
| **10** | **4.55** | **4.50** | **89.63** |

---

## v3 Evaluation Results

### Overall Score
**16.4% keyword match** (50 questions)

### By Category

| Category | v1 | v3 | Change |
|----------|----|----|--------|
| basics | 32.6% | 32.4% | -0.2% |
| entanglement | 9.0% | 27.0% | **+18.0%** |
| superposition | 9.0% | 20.7% | **+11.7%** |
| measurement | 4.2% | 11.1% | **+6.9%** |
| gates | 20.8% | 15.0% | -5.8% |
| algorithms | 18.0% | 13.0% | -5.0% |
| hardware | 8.0% | 4.0% | -4.0% |
| applications | 6.7% | 6.7% | 0% |

### By Difficulty

| Difficulty | v3 Score | Count |
|------------|----------|-------|
| easy | 23.0% | 16 |
| medium | 14.7% | 26 |
| hard | 8.8% | 8 |

### Quality Flags

| Metric | v1 | v3 |
|--------|----|----|
| Repetitive outputs | Many | 2 |
| Boilerplate phrases | Heavy | None |

---

## v3 Verification Results

### Boilerplate Detection

| Metric | v1 | v3 |
|--------|----|----|
| Boilerplate phrases | 83.4% | **0.0%** |
| Template patterns | 59.0% | **0.0%** |

**Result:** SUCCESS - No contamination detected in v3 outputs.

### Checkpoint Comparison (Epoch 1 → 5 → 10)

| Question | Epoch 1 | Epoch 5 | Epoch 10 |
|----------|---------|---------|----------|
| What is a qubit? | "ﬁnal(." | Forming sentences | Complete sentences |
| What is superposition? | "maybe a yes the a uniti" | Broken but quantum terms | Proper terminology |
| What is entanglement? | Garbage fragments | Some structure | Coherent attempts |

**Result:** Clear progression across epochs. Training worked correctly.

---

## Final Assessment

| Aspect | v1 (Garbage) | v3 (Clean) | Status |
|--------|--------------|------------|--------|
| Boilerplate phrases | 83.4% | 0% | ✅ Fixed |
| Template patterns | 59.0% | 0% | ✅ Fixed |
| Keyword match | 14.8% | 16.4% | ✅ Improved |
| Repetitive flags | Many | 2 | ✅ Fixed |
| Coherent reasoning | No | No | Expected (1.2M params) |

**Conclusion:** Data cleaning worked. Model trained correctly. Limited by size. RAG essential for usability.

---

## Final Dataset Composition

| Source | Count | Est. Tokens | Status |
|--------|-------|-------------|--------|
| Claude Q&A | 15,000 pairs | ~2.3M | ✅ Complete |
| Stack Exchange (filtered) | 9,019 pairs | ~1.2M | ✅ Complete |
| Books | 633,562 words | ~0.9M | ✅ Complete |
| **Total** | **24,019 Q&A** | **~4.4M** | ✅ Trained |

---

## What Is Next

**Immediate next task:** Phase 2 - RAG System

### Phase 2: RAG System

| Task | Priority | Status |
|------|----------|--------|
| Chunk books for RAG (~500 tokens, semantic) | High | ⬜ Pending |
| Generate embeddings (Voyage AI) | High | ⬜ Pending |
| Set up Neon database with pgvector | High | ⬜ Pending |
| Implement retrieval pipeline | High | ⬜ Pending |
| Test retrieval quality | Medium | ⬜ Pending |

---

## Output Files on HPC

| File | Location | Description |
|------|----------|-------------|
| `final_model.pt` | `model/` | Final model weights (v3, epoch 10) |
| `best_model.pt` | `model/` | Best model by val loss |
| `checkpoint_epoch[1-10].pt` | `model/` | All epoch checkpoints |
| `config.json` | `model/` | Model config |
| `combined_qa_filtered.csv` | `data/` | 24,019 Q&A pairs |
| `tokenizer.json` | `/` | BPE tokenizer (16K vocab) |
| `evaluation_results.json` | `scripts/` | Full eval results |

---

## Development Phases

### Phase 1: Training Pipeline ✅ COMPLETE

| Task | Status |
|------|--------|
| Generate Claude Q&A (15,000 pairs) | ✅ Done |
| Combine final dataset | ✅ Done |
| Train model v3 (10 epochs) | ✅ Done |
| Evaluate model (16.4%) | ✅ Done |
| Verify quality (0% boilerplate) | ✅ Done |

### Phase 2: RAG System ⬜ NEXT

| Task | Status |
|------|--------|
| Chunk books for RAG | ⬜ Pending |
| Generate embeddings (Voyage AI) | ⬜ Pending |
| Set up Neon database with pgvector | ⬜ Pending |
| Implement retrieval pipeline | ⬜ Pending |
| Test retrieval quality | ⬜ Pending |

### Phase 3: Backend

| Task | Status |
|------|--------|
| FastAPI endpoints | ⬜ Pending |
| Custom model inference | ⬜ Pending |
| RAG integration | ⬜ Pending |

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

## Key Findings During Implementation

1. **ChatGPT synthetic data was 94% garbage.** Boilerplate in 83%, templates in 59%. Abandoned entirely.

2. **Claude Q&A generation works.** 15,000 pairs with 100% unique questions, proper verification.

3. **More epochs help.** Checkpoint comparison showed clear progression epoch 1 → 5 → 10.

4. **Higher perplexity can be better.** v1's low perplexity came from memorizing garbage.

5. **Small models learn vocabulary, not reasoning.** 1.2M params produces quantum jargon but incoherent answers. RAG essential.

6. **Data quality verification is critical.** Boilerplate check confirmed 0% contamination in v3.

7. **H100s are fast.** 10 epochs in 13 minutes at 620K tokens/sec.

---

*Document version: 9.0*
