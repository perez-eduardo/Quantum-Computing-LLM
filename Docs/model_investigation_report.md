# Model Investigation Report

**Date:** December 21, 2025
**Updated:** December 24, 2025
**Purpose:** Document findings from model training, evaluation, and RAG system

---

## Current Status

**Training:** ⚠️ Requires retraining (context-aware format)
**RAG:** ✅ Complete (94% retrieval)
**Data Processing:** ✅ Complete (context added to all datasets)
**Next:** Retrain model with context-aware format (50M-100M params)

---

## CRITICAL DISCOVERY: Design Flaw (December 24, 2025)

### The Problem

v1-v4 models were trained on plain Q&A format:
```
Q: What is superposition?
A: Superposition allows...
```

At inference, RAG retrieves context:
```
Context: [retrieved Q&A pairs]
Question: What is superposition?
```

**The model ignores context because it never learned the format.**

Additionally, 1.2M parameters cannot produce coherent sentences - only quantum jargon in gibberish structure.

### The Solution

1. Retrain with context-aware format
2. Scale to 50M-100M parameters
3. Model learns: Context + Question → Answer

---

## Training Results Comparison (Pre-Redesign)

| Metric | v1 (Garbage) | v3 (Clean) | v4 (Expanded) |
|--------|--------------|------------|---------------|
| Training data | 96K (94% garbage) | 24K | 26,764 |
| Format | Plain Q&A | Plain Q&A | Plain Q&A |
| Epochs | 3 | 10 | 10 |
| Duration | ~4 min | ~13 min | ~13 min |
| Final Perplexity | 15.55 | 89.63 | 91.80 |
| Eval Score | 14.8% | 16.4% | 11.4% |
| Boilerplate | 83.4% | 0% | 0% |
| Can use RAG context | ❌ No | ❌ No | ❌ No |

**All versions are fundamentally limited** - cannot use RAG context at inference.

---

## Data Processing Complete (December 24, 2025)

### Context-Format Datasets Created

| File | Rows | Context Source |
|------|------|----------------|
| cot_qa_context.csv | 2,998 | Reasoning from metadata field |
| stackexchange_qa_context.csv | 10,673 | Tags + question body |
| claude_qa_batch1-38_context.csv | 15,000 | Template-based (question type + topics) |
| **Total** | **28,671** | |

### CoT Dataset Context

Original JSON had `metadata.reasoning` field containing chain-of-thought:
```json
{
  "question": "Explain wave-particle duality...",
  "answer": "Wave-particle duality is...",
  "metadata": {
    "reasoning": "My approach begins with breaking down the definition..."
  }
}
```

Extracted reasoning as context column.

### Stack Exchange Context

Combined tags + question body:
```
Topics: entanglement, linear-algebra. The Bell state |Φ+⟩ = 1/√2(|00⟩ + |11⟩) is an entangled state. But why is that the case?
```

### Claude Q&A Context

Template-based context generation:
- Question type detection (definitional, procedural, causal, comparison)
- Topic keyword extraction (qubit, entanglement, gates, algorithms, etc.)
- Answer complexity assessment

Example:
```
This definitional question requires a clear explanation of the fundamental concept. The answer draws on knowledge of quantum bits and their properties, quantum gate operations. A balanced explanation covers the key points concisely.
```

---

## RAG System Results (December 24, 2025)

### Retrieval Quality: 94%

| Version | Contents | Pass Rate |
|---------|----------|-----------|
| v1 | 2,847 book chunks | 92% |
| v2 | Books + 26,764 Q&A | 94% |
| v2 (final) | 26,764 Q&A only | 94% |

### Current Database Contents

| Source | Count |
|--------|-------|
| claude_synthetic | 15,000 |
| stackexchange | 9,008 |
| cot_reasoning | 2,756 |
| **Total** | **26,764** |

### Remaining Failures (6/100)

| Query | Issue Type |
|-------|------------|
| Computational basis | Retrieved question instead of definition |
| Fredkin gate | Data gap (not in training data) |
| QAOA | Acronym confusion with QBism |
| Quantum counting | Retrieved excerpt, not definition |
| Partial trace | Retrieved related content, not definition |
| Fidelity | Retrieved "layer fidelity" variant |

---

## v4 Training Details (December 24, 2025)

### Dataset Composition

| Source | Count |
|--------|-------|
| Claude Q&A | 15,000 |
| Stack Exchange (filtered) | 9,008 |
| CoT Reasoning Dataset | 2,756 |
| **Total** | **26,764** |

### Training Results

| Metric | Value |
|--------|-------|
| Job ID | On dgxh (H100 80GB) |
| Epochs | 10 |
| Final Perplexity | 91.80 |
| Eval Score | 11.4% keyword match |
| Boilerplate | 0% |

### Model Files (Downloaded but OUTDATED)

| File | Status |
|------|--------|
| `final_model.pt` | ❌ Cannot use context |
| `best_model.pt` | ❌ Cannot use context |
| `config.json` | ⚠️ Needs update for 50M-100M |
| `tokenizer.json` | ✅ Still valid |

---

## v3 Evaluation Results (December 24, 2025)

### Overall Score
**16.4% keyword match** (50 questions)

### By Category

| Category | v1 | v3 | Change |
|----------|----|----|--------|
| basics | 32.6% | 32.4% | -0.2% |
| entanglement | 9.0% | 27.0% | **+18.0%** ✓ |
| superposition | 9.0% | 20.7% | **+11.7%** ✓ |
| measurement | 4.2% | 11.1% | **+6.9%** ✓ |
| gates | 20.8% | 15.0% | -5.8% |
| algorithms | 18.0% | 13.0% | -5.0% |
| hardware | 8.0% | 4.0% | -4.0% |
| applications | 6.7% | 6.7% | 0% |

---

## v1 Investigation (December 21, 2025)

### Root Cause

ChatGPT synthetic Q&A data was 94% garbage:
- 83% contained repetitive boilerplate
- 59% were templated (only numbers changed)
- Only 4,808 usable from 85,643 pairs

**Decision:** Abandoned ChatGPT data entirely. Replaced with Claude-generated Q&A.

---

## Next Steps: Model v5

### Requirements

1. **Context-aware training format:**
```
Context:
Q: [context pair 1]
A: [answer 1]

Q: [context pair 2]
A: [answer 2]

Question: [target question]
Answer: [target answer]
```

2. **Larger architecture:** 50M-100M parameters for coherent generation

3. **Training data:** 28,671 context-format Q&A pairs ready

### Infrastructure

| Resource | Status |
|----------|--------|
| HPC access | ✅ Available |
| Training data | ✅ Ready (context format) |
| Model architecture | ⬜ Needs update |
| Training time estimate | 30-60 min on H100 |

---

## Lessons Learned

1. **Don't trust synthetic data blindly.** ChatGPT generated 94% garbage despite careful prompting.

2. **Training format must match inference format.** Plain Q&A model cannot use RAG context.

3. **Small models learn vocabulary, not reasoning.** 1.2M params produces quantum jargon but incoherent answers.

4. **Coherent generation requires capacity.** Need 50M-100M parameters for readable output.

5. **Inspect data at every step.** Initial "clean" data had massive hidden issues.

6. **Higher perplexity can be better.** Low perplexity from garbage means memorization, not learning.

7. **Q&A pairs beat book chunks for RAG.** Definitions > mentions.

8. **94% retrieval is achievable and sufficient.** Remaining failures are edge cases.

9. **Verify the complete pipeline before deployment.** Discovered design flaw at implementation phase.

---

## Files

### Context-Format Data (NEW)

| File | Rows | Location |
|------|------|----------|
| cot_qa_context.csv | 2,998 | outputs/ |
| stackexchange_qa_context.csv | 10,673 | outputs/ |
| claude_qa_batch1-38_context.csv | 15,000 | outputs/ |

### Local (training/model/) - OUTDATED

| File | Description |
|------|-------------|
| `final_model.pt` | v4 model weights (cannot use context) |
| `best_model.pt` | Best model by val loss |
| `config.json` | Model config (needs update) |
| `tokenizer.json` | BPE tokenizer (16K vocab) - still valid |

### Database (Neon)

| Table | Contents |
|-------|----------|
| chunks | 26,764 Q&A embeddings |

---

*Document version: 9.0*
*Last updated: December 24, 2025*
