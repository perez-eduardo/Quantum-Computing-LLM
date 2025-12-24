# Model Investigation Report

**Date:** December 21, 2025
**Updated:** December 24, 2025
**Purpose:** Document findings before deciding on retraining strategy

> **✅ FINAL OUTCOME (December 24, 2025)**
> 
> - ChatGPT data was 94% garbage → **Abandoned**
> - Claude Q&A generation **COMPLETE**: 15,000 pairs across 38 batches
> - Model v3 **TRAINED**: 10 epochs, perplexity 89.63
> - **Next step:** Evaluation

---

## Training Results Comparison

### v1 (Garbage Data) vs v3 (Clean Data)

| Metric | v1 (Dec 21) | v3 (Dec 23) |
|--------|-------------|-------------|
| Data | 96K Q&A (94% ChatGPT garbage) | 24K Q&A (Claude + SE) |
| Epochs | 3 | 10 |
| Duration | ~4 min | ~13 min |
| Final Val Loss | 2.74 | 4.50 |
| Final Perplexity | 15.55 | 89.63 |
| Eval Accuracy | 14.8% | TBD |

**Why v3 perplexity is higher:** v1's low perplexity came from memorizing repetitive garbage (same boilerplate phrases over and over). v3 trained on diverse, clean data has higher perplexity but should produce coherent outputs. The real test is evaluation.

---

## v1 Model Performance (Garbage Data)

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

**Qualitative Observations (Sanity Check):**
- Outputs contain quantum terminology
- Responses are incoherent fragments stitched together
- Model rambles, doesn't answer the actual question
- Broken LaTeX, trails off mid-sentence
- Stays loosely in domain but no reasoning

---

## v3 Training Results (Clean Data)

**Job:** 19759979 on dgxh-1 (H100 80GB)
**Duration:** ~13 minutes
**Data:** 24,019 Q&A pairs + books
**Throughput:** ~620K tokens/sec

### Loss Progression

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

### Training Config (v3)

| Parameter | v1 | v3 | Change |
|-----------|----|----|--------|
| Epochs | 3 | 10 | +7 |
| Batch size | 64 | 64 | - |
| Max LR | 3e-4 | 3e-4 | - |
| Min LR | 3e-5 | 1e-5 | Lower |
| Warmup ratio | 0.1 | 0.05 | Lower |
| Max seq length | 512 | 512 | - |

### Key Observations

1. **Steady convergence** - Loss decreased consistently across all 10 epochs
2. **No overfitting** - Val loss tracks train loss throughout
3. **Diminishing returns** - Most improvement in epochs 1-6, slower after
4. **Final perplexity 89.63** - Higher than v1 but expected for diverse data

---

## v1 Training Log Analysis

**Job:** 19739587 on dgxh-1 (H100 80GB)
**Duration:** ~4 minutes
**Throughput:** ~626K tokens/sec

### Loss Progression
| Epoch | Train Loss | Val Loss | Perplexity |
|-------|------------|----------|------------|
| 1 | 5.32 | 3.23 | 25.29 |
| 2 | 2.94 | 2.82 | 16.84 |
| 3 | 2.74 | 2.74 | 15.55 |

### Key Observations
1. **Loss still decreasing** at epoch 3 - model not converged
2. **Perplexity still improving** - would likely keep dropping
3. **Only 4,461 steps** - very short training
4. **LR schedule:** Started 3e-4, ended at 3e-5 (cosine decay bottomed out)
5. **No overfitting** - val loss tracks train loss

---

## Token Length Analysis (Original Data)

**Dataset:** 96,305 Q&A pairs (before cleaning)

### Combined Tokens (Q + A + formatting)
| Metric | Value |
|--------|-------|
| Min | 29 |
| Max | 13,185 |
| Mean | 138.6 |
| Median | 76 |
| P95 | 553 |
| P99 | 1,260 |

### Length Distribution
| Bucket | Count | Percent |
|--------|-------|---------|
| 0-128 | 85,259 | 88.5% |
| 129-256 | 2,035 | 2.1% |
| 257-384 | 1,980 | 2.1% |
| 385-512 | 1,741 | 1.8% |
| 513+ (truncated) | 5,290 | 5.5% |

---

## Identified Issues (v1)

### Issue 1: Undertrained
- Loss still dropping at epoch 3
- Perplexity would keep improving with more training
- LR decayed to minimum before convergence
- **Checkpoint comparison confirms:** Epoch 6 >> Epoch 3 >> Epoch 1

### Issue 2: Data Imbalance
- 88.5% of training examples are under 128 tokens
- Median is only 76 tokens
- Model mostly sees very short Q&A pairs
- May not learn complex reasoning from brief examples

### Issue 3: Extreme Outliers
- Some examples are 13,000+ tokens
- These get brutally truncated to 512
- Model sees question + start of answer, never the conclusion
- Potentially confusing training signal

### Issue 4: Repetitive Boilerplate in Training Data
- Model memorized template phrases from ChatGPT synthetic data
- Every answer ends with "In fault-tolerant settings..." or similar
- Model does pattern completion, not Q&A reasoning

---

## Checkpoint Comparison Results (v1)

Tested epochs 1, 3, and 6 (from cancelled job that ran 6 epochs).

| Epoch | Quality | Coherence | Boilerplate |
|-------|---------|-----------|-------------|
| 1 | Worst | Fragments, broken LaTeX | Present |
| 3 | Medium | Some complete sentences | Heavy |
| 6 | Best | Good sentences, some correct answers | Heavy |

**Conclusion:** Model improves with more epochs but has memorized repetitive patterns from training data.

---

## Summary of Issues Found

| Issue | Severity | Action |
|-------|----------|--------|
| Boilerplate in 83% of ChatGPT data | Critical | ❌ Data abandoned |
| 59% templated repetitive examples | Critical | ❌ Data abandoned |
| 5.5% truncated examples (code dumps) | High | ✅ Filtered from Stack Exchange |
| Undertrained (3 epochs) | Medium | ✅ Used 10 epochs in v3 |

---

## Final Decision (December 22, 2025)

### Decision: Abandon ChatGPT Data

The ChatGPT synthetic Q&A data is unsalvageable:

| Finding | Value |
|---------|-------|
| Templated questions (only numbers changed) | **59%** |
| Boilerplate phrases | 83% |
| Usable rows after all cleaning | **4,808 of 85,643** |
| **Garbage rate** | **94%** |

---

## New Data Strategy

### Claude Q&A Generation - COMPLETE ✅

**Completed:** December 23, 2025

| Metric | Value |
|--------|-------|
| Total Q&A pairs | **15,000** |
| Batches | 38 (Phase 1: 25 × 400, Phase 2: 12 × 400 + 1 × 200) |
| Unique questions | 15,000 (100%) |
| Duplicates fixed | ~250 across all batches |
| Generation method | 8-chunk verification per batch |
| Output files | `claude_qa_batch[1-38].csv` |

**Topics Covered (38 Batches):**

**Phase 1 - Topic-Based (Batches 1-25):**
| Batch | Topic |
|-------|-------|
| 1-5 | Fundamentals, Algorithms, Hardware, Error Correction, Chemistry |
| 6-10 | ML, Cryptography, Complexity, Many-body, Topological |
| 11-15 | Simulation, Annealing, Control, Metrology, Sensing |
| 16-20 | Thermodynamics, Foundations, Optics, Superconducting, Communication |
| 21-22 | Trapped Ion, Neutral Atom |
| 23 | Photonic Quantum Computing |
| 24 | Quantum Software and Programming |
| 25 | Quantum Applications and Industry |

**Phase 2 - Question Format-Based (Batches 26-38):**
| Batch | Question Format |
|-------|-----------------|
| 26 | "What/How does X work" (mechanisms) |
| 27 | "Why" questions (reasoning and causes) |
| 28 | "What happens when/if" (consequences) |
| 29 | Troubleshooting and problem-solving |
| 30 | Best practices and recommendations |
| 31 | Definitions and explanations |
| 32 | "When/What if" conditional questions |
| 33 | "Can/Could" possibility questions |
| 34 | "Is it true that" fact-checking |
| 35 | "Which" choice and selection |
| 36 | Comparisons (X vs Y) |
| 37 | "Should I" recommendations |
| 38 | Mixed final questions |

### Final Dataset Composition

| Source | Count | Est. Tokens | Status |
|--------|-------|-------------|--------|
| Claude Q&A | 15,000 pairs | ~2.3M | ✅ Complete |
| Stack Exchange (filtered) | 9,019 pairs | ~1.2M | ✅ Complete |
| Books | 633,562 words | ~0.9M | ✅ Complete |
| **Total** | **24,019 Q&A** | **~4.4M** | |

---

## Next Steps

1. ✅ Clean Stack Exchange data (filtered >1024 tokens)
2. ✅ Generate Claude Q&A (15,000 pairs)
3. ✅ Combine final dataset (24,019 Q&A pairs)
4. ✅ Retrain tokenizer on new corpus
5. ✅ Retrain model with 10 epochs
6. ⬜ **Evaluate model** ← Next
7. ⬜ Compare v1 vs v3 outputs
8. ⬜ Document for portfolio

---

## Lessons Learned

1. **Don't trust synthetic data blindly.** ChatGPT generated 94% garbage despite careful prompting.
2. **Inspect data at every step.** Initial "clean" data had massive hidden issues.
3. **Template detection requires statistical analysis.** 30% estimate was off by 2x.
4. **Boilerplate removal alone is insufficient.** Underlying data quality matters more.
5. **Sometimes it's better to start fresh.** Salvaging bad data wastes time.
6. **Chunk-based generation with verification works.** 8-chunk batches with incremental duplicate checking caught issues early.
7. **Index-based duplicate detection scales.** Maintaining a questions index enabled cross-batch duplicate detection.
8. **Diverse question formats improve coverage.** Supplementing topic-based with format-based questions creates more natural variety.
9. **Higher perplexity can be better.** Low perplexity from garbage data means memorization, not learning.
10. **Don't load system Python after venv activation.** Module load overrides venv and breaks imports.

---

*Document version: 5.0*
*Last updated: December 24, 2025*
