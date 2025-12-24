# Quantum Computing Assistant - Implementation Plan

**Last Updated:** December 24, 2025

**Related Documents:**
- Design Document: `quantum-computing-assistant-design.md`
- Infrastructure Planning: `initial-exploratory-brainstorming.md`
- Model Investigation Report: `model_investigation_report.md`

---

## Current Status

**Phase:** 1 (Training Pipeline) - 🔄 REDO REQUIRED
**Status:** New data source identified. Retraining with expanded dataset.

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
| **Obtain CoT Reasoning Dataset** | ✅ Done | **3,000 Q&A pairs with chain-of-thought** |

### Data Processing & Cleaning

| Task | Status | Output |
|------|--------|--------|
| Process Stack Exchange XML | ✅ Done | 10,673 pairs |
| Filter Stack Exchange (>1024 tokens) | ✅ Done | 9,019 pairs |
| Extract and clean book texts | ✅ Done | 633,562 words |
| Generate Claude Q&A | ✅ Done | 15,000 pairs |
| **Process CoT dataset** | ⬜ Pending | 3,000 pairs |
| **Combine all sources** | ⬜ Pending | ~27,019 Q&A pairs expected |
| Train custom BPE tokenizer | ✅ Done | 16K vocab (may need retrain) |

### HPC Training

| Task | Status | Notes |
|------|--------|-------|
| Set up HPC environment | ✅ Done | Python 3.11 venv, PyTorch 2.5.1+cu121 |
| Implement transformer architecture | ✅ Done | `scripts/model.py` |
| Train model v1 (garbage data) | ✅ Done | 3 epochs, perplexity 15.55, 14.8% eval |
| Investigate data quality issues | ✅ Done | ChatGPT data 94% garbage |
| Train model v3 (clean data) | ✅ Done | 10 epochs, perplexity 89.63 |
| Evaluate model v3 | ✅ Done | 16.4% keyword match |
| Verify data quality | ✅ Done | 0% boilerplate contamination |
| **Train model v4 (expanded data)** | ⬜ Pending | **Include CoT dataset** |

---

## New Data Source

### CoT_Reasoning_Quantum_Physics_And_Computing.json

| Property | Value |
|----------|-------|
| Location | `data/raw/source/CoT_Reasoning_Quantum_Physics_And_Computing.json` |
| Total entries | 3,000 Q&A pairs |
| Answer length | ~3,000-4,000 chars each |
| Structure | question, answer, metadata (topic, difficulty, reasoning) |
| License | MIT (open source) |

**Why include this:**
- Pre-structured Q&A format
- Chain-of-thought reasoning included
- Self-contained explanations
- Covers wide range: fundamentals, algorithms, hardware, sensors, QED/QCD

**Sample topics:**
- Wave-particle duality
- Quantum sensors
- POVMs
- Semiconductor spin qubits
- QED vs QCD

---

## Revised Dataset Composition

### v4 Training Data (Planned)

| Source | Count | Est. Tokens | Status |
|--------|-------|-------------|--------|
| Claude Q&A | 15,000 pairs | ~2.3M | ✅ Ready |
| Stack Exchange (filtered) | 9,019 pairs | ~1.2M | ✅ Ready |
| **CoT Reasoning Dataset** | **3,000 pairs** | **~1.5M** | ✅ Ready |
| Books | 633,562 words | ~0.9M | ✅ Ready |
| **Total** | **~27,019 Q&A** | **~5.9M** | ⬜ Combine |

### Comparison

| Version | Q&A Pairs | Est. Tokens |
|---------|-----------|-------------|
| v3 | 24,019 | ~4.4M |
| v4 (planned) | ~27,019 | ~5.9M |

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

### v4 (Expanded Data - Planned)

| Metric | Value |
|--------|-------|
| Data | ~27K Q&A (Claude + Stack Exchange + CoT) |
| Epochs | TBD |
| Perplexity | TBD |
| Eval Score | TBD |
| Boilerplate | TBD |

---

## What Is Next

**Immediate next task:** Prepare v4 training data

### Phase 1 (Redo): Training Pipeline

| Task | Priority | Status |
|------|----------|--------|
| Process CoT dataset into training format | High | ⬜ Pending |
| Combine all Q&A sources | High | ⬜ Pending |
| Verify combined dataset quality | High | ⬜ Pending |
| Retrain tokenizer (if needed) | Medium | ⬜ Pending |
| Train model v4 | High | ⬜ Pending |
| Evaluate model v4 | High | ⬜ Pending |

### Phase 2: RAG System

| Task | Priority | Status |
|------|----------|--------|
| Chunk all sources for RAG | High | ⬜ Pending |
| Generate embeddings (Voyage AI) | High | ⬜ Pending |
| Set up Neon database with pgvector | High | ✅ Done |
| Implement retrieval pipeline | High | ⬜ Pending |
| Test retrieval quality | High | ⬜ Pending |

**RAG Data Sources:**
1. 5 books (chunked)
2. Stack Exchange Q&A
3. CoT Reasoning Dataset Q&A

---

## Output Files on HPC

| File | Location | Description |
|------|----------|-------------|
| `final_model.pt` | `model/` | v3 model weights (to be replaced by v4) |
| `best_model.pt` | `model/` | Best model by val loss |
| `checkpoint_epoch[1-10].pt` | `model/` | v3 checkpoints |
| `config.json` | `model/` | Model config |
| `combined_qa_filtered.csv` | `data/` | v3 training data (24,019 pairs) |
| `tokenizer.json` | `/` | BPE tokenizer (16K vocab) |
| `evaluation_results.json` | `scripts/` | v3 eval results |

---

## Development Phases

### Phase 1: Training Pipeline 🔄 REDO

| Task | Status |
|------|--------|
| Generate Claude Q&A (15,000 pairs) | ✅ Done |
| Process Stack Exchange | ✅ Done |
| **Process CoT dataset (3,000 pairs)** | ⬜ Pending |
| **Combine all sources** | ⬜ Pending |
| **Train model v4** | ⬜ Pending |
| **Evaluate model v4** | ⬜ Pending |

### Phase 2: RAG System ⬜ NEXT

| Task | Status |
|------|--------|
| Chunk all sources for RAG | ⬜ Pending |
| Generate embeddings (Voyage AI) | ⬜ Pending |
| Set up Neon database with pgvector | ✅ Done |
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

8. **CoT dataset provides structured reasoning.** 3,000 pairs with chain-of-thought explanations.

---

*Document version: 10.0*
