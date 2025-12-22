# Model Investigation Report

**Date:** December 21, 2025
**Purpose:** Document findings before deciding on retraining strategy

---

## Current Model Performance

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

## Training Log Analysis

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

### Training Config Used
| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size | 64 |
| Max LR | 3e-4 |
| Min LR | 3e-5 |
| Warmup ratio | 0.1 |
| Max seq length | 512 |

---

## Token Length Analysis

**Dataset:** 96,305 Q&A pairs

### Question Tokens
| Metric | Value |
|--------|-------|
| Min | 5 |
| Max | 13,064 |
| Mean | 47.4 |
| Median | 19 |
| P95 | 207 |
| P99 | 589 |

### Answer Tokens
| Metric | Value |
|--------|-------|
| Min | 12 |
| Max | 8,357 |
| Mean | 89.2 |
| Median | 55 |
| P95 | 286 |
| P99 | 812 |

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
| 513-768 (truncated) | 2,408 | 2.5% |
| 769-1024 (truncated) | 1,248 | 1.3% |
| 1025+ (truncated) | 1,634 | 1.7% |

### Truncation
- **5,290 pairs (5.5%)** exceed 512 tokens
- Some examples are 12,000+ tokens over limit (code dumps from Stack Exchange)
- Mean overage: 467 tokens

---

## Identified Issues

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

### Issue 4: max_seq_len Mismatch
- 512 tokens allocated
- 88.5% of data uses less than 128
- Wasted capacity? Or fine since padding is ignored?

### Issue 5: Repetitive Boilerplate in Training Data (NEW)
- Model memorized template phrases from ChatGPT synthetic data
- Every answer ends with "In fault-tolerant settings..." or similar
- Model does pattern completion, not Q&A reasoning
- Need to inspect and possibly clean ChatGPT data

---

## Checkpoint Comparison Results

Tested epochs 1, 3, and 6 (from cancelled job that ran 6 epochs).

### Epoch 1 Output (Worst)
```
Q: What is a qubit?
A: In quantum computing, probabilities of coherence, gate speed, connectivity, and engineering interference.

Q: What is entanglement?
A: I can be the state is a state. In

Q: How is a qubit different from a classical bit?
A: |0\rangle, \left_{1\rangle}
\rangle \frac{1\rangle) for the circuit, with other state, and $
```
- Broken LaTeX
- Incomplete sentences
- Barely coherent fragments

### Epoch 3 Output (Better)
```
Q: What is a CNOT gate?
A: CNOT flips the target qubit if the control is |1⟩; it is widely used to create entanglement and implement parity checks. If the device lacks a direct interaction needed, compilation may add routing (SWAPs) and express it in native gates, increasing depth and error.
```
- More complete sentences
- Some answers almost correct
- Still rambles with boilerplate

### Epoch 6 Output (Best)
```
Q: What is a qubit?
A: A qubit is the Bloch sphere that becomes visible through interference. In fault-tolerant settings, limited connectivity can force extra SWAP routing.

Q: What is a CNOT gate?
A: CNOT flips the target qubit if the control is |1⟩; it is widely used to create entanglement and implement parity checks. For intuition, think of it as a rotation/phase action on the Bloch sphere that becomes visible through interference.
```
- Cleaner output
- CNOT answer is actually good
- Still appends irrelevant boilerplate

### Key Finding: Repetitive Boilerplate

Almost every answer contains one of these phrases:
- "In fault-tolerant settings..."
- "In practical workflows..."
- "In NISQ experiments..."
- "Don't confuse this with classical parallelism..."
- "...error mitigation may reduce bias but doesn't fully protect quantum information"

**Likely cause:** Synthetic ChatGPT training data used templates/boilerplate that the model memorized.

### Checkpoint Comparison Summary

| Epoch | Quality | Coherence | Boilerplate |
|-------|---------|-----------|-------------|
| 1 | Worst | Fragments, broken LaTeX | Present |
| 3 | Medium | Some complete sentences | Heavy |
| 6 | Best | Good sentences, some correct answers | Heavy |

**Conclusion:** Model improves with more epochs but has memorized repetitive patterns from training data.

---

## Truncated Examples Analysis

### Summary
- **5,290 examples (5.5%)** exceed 512 token limit
- **100% from Stack Exchange** (none from ChatGPT)

### Truncation Severity
| Tokens Over Limit | Count | Percent |
|-------------------|-------|---------|
| 1-100 | 1,120 | 21.2% |
| 101-500 | 2,489 | 47.1% |
| 501-1000 | 1,103 | 20.9% |
| 1001-5000 | 569 | 10.8% |
| 5000+ | 9 | 0.2% |

### Content Analysis
- **31.1%** contain code blocks
- Average tokens over limit: 467
- **31.8% severely truncated** (>500 tokens cut)

### What Gets Truncated
Examples are mostly garbage:
- Code dumps (Qiskit circuits, installation logs)
- Configuration outputs (JSON, package lists)
- Long stack traces
- Raw data printouts

Model sees question + start of answer, but the actual solution gets cut off.

### Recommendation
Filter out examples >1024 tokens before training. They provide confusing signal (partial answers with no conclusion).

---

## Short Examples Analysis

### Summary
- **29.9% very short** (<50 tokens): 28,823 examples
- **59.1% short** (50-128 tokens): 56,918 examples
- **88% of data is under 128 tokens**

### Source Breakdown
| Category | ChatGPT | Stack Exchange |
|----------|---------|----------------|
| Very short | 100% | 0% |
| Short | 99.6% | 0.4% |

### Quality Issue: Templated Repetitive Examples

The "very short" examples are templated garbage with only numbers changed:

```
Q: Why does the state space grow exponentially with 198 qubits?
A: Because each additional qubit doubles the dimension... 2^198 basis states

Q: Why does the state space grow exponentially with 319 qubits?
A: Because each additional qubit doubles the dimension... 2^319 basis states

Q: Why does the state space grow exponentially with 294 qubits?
A: Because each additional qubit doubles the dimension... 2^294 basis states
```

Same templates repeated hundreds of times:
- "Why does the state space grow exponentially with X qubits?"
- "How many complex amplitudes describe a generic X-qubit pure state?"
- "How many basis states are there for X qubits?"

Model is memorizing templates, not learning concepts.

### Recommendation
Filter out templated repetitive examples. Identify templates and keep only 1-2 examples of each pattern instead of hundreds.

---

## Book Data Contribution

### Summary
- **980,964 book tokens** across 5 textbooks
- **3,830 book chunks** (512 tokens, stride 256)
- **3.8% of training examples** (25:1 ratio Q&A to book)

### Token Contribution
| Source | Tokens | Percent |
|--------|--------|---------|
| Q&A (est) | 7,314,620 | 88.2% |
| Books | 980,964 | 11.8% |

### Book Sources
1. Quantum Computing Explained for Beginners (Pantheon Space Academy)
2. Quantum Computing for Everyone (Chris Bernhardt, MIT Press)
3. Quantum Computing: An Applied Approach (Jack Hidary)
4. Quantum Computation and Quantum Information (Nielsen & Chuang)
5. Introduction to Classical and Quantum Computing (Thomas Wong)

### Content Quality
- 72,789 lines
- 23% short lines (<20 chars) - likely headers, equations
- 0.1% math notation lines

### Recommendation
Book data is drowned out at 3.8%. Filtering Q&A templates will naturally improve balance. No need to upsample books.

---

## Summary of Issues Found

| Issue | Severity | Action |
|-------|----------|--------|
| Boilerplate in 83% of ChatGPT data | ✅ Fixed | Cleaned, 0% remaining |
| 5.5% truncated examples (code dumps) | High | Filter >1024 tokens |
| 30% templated repetitive examples | High | Deduplicate templates |
| Book data only 3.8% | Medium | Will improve after Q&A filtering |
| Undertrained (3 epochs) | Medium | Increase to 10 epochs |

## Recommended Actions

1. Filter examples >1024 tokens
2. Deduplicate templated Q&A examples
3. Retrain with cleaned data, 10 epochs

---

## Data Cleaning Performed

### Boilerplate Analysis Results
- **83.4%** of ChatGPT answers contained boilerplate
- **65.9%** had 2+ boilerplate phrases
- Common patterns appended to answers:
  - "In fault-tolerant settings, [boilerplate]"
  - "In practical workflows, [boilerplate]"
  - "In NISQ experiments, [boilerplate]"
  - "In many tutorials, [boilerplate]"
  - "In simulators, [boilerplate]"
  - "On real devices, [boilerplate]"

### Cleaning Action
Ran `clean_boilerplate.py` to strip boilerplate phrases from answers.

| Metric | Before | After |
|--------|--------|-------|
| Total rows | 85,643 | 85,583 |
| With boilerplate | 71,427 (83.4%) | 7 (0.0%) |
| Removed (too short) | - | 60 |

### New Combined Dataset
- File: `combined_qa_final_v2.csv` (replaced old on HPC)
- Total rows: 96,245
- Estimated tokens: ~12.4M (down from ~14.2M)
- Boilerplate remaining: 7/96,245 (0.0%)

---

## Next Steps

1. ~~Compare checkpoint outputs (epoch 1 vs 2 vs 3)~~ ✅ Done
2. ~~Inspect ChatGPT training data for boilerplate patterns~~ ✅ Done
3. ~~Clean boilerplate from training data~~ ✅ Done
4. ~~Upload cleaned data to HPC~~ ✅ Done
5. Filter truncated examples (>1024 tokens)
6. Deduplicate templated examples
7. Retrain model with clean data

---

## Consolidated Recommendations

### Data Cleaning (Before Retraining)

| Action | Reason | Est. Rows Affected |
|--------|--------|-------------------|
| ✅ Remove boilerplate | Model memorized templates | 71,427 cleaned |
| Filter >1024 tokens | Truncated garbage (code dumps, logs) | ~1,700 removed |
| Deduplicate templates | Same Q with different numbers | ~20,000+ removed |

### Training Hyperparameters

| Parameter | Old | New | Reason |
|-----------|-----|-----|--------|
| Epochs | 3 | 10 | Loss still dropping at epoch 3, checkpoint 6 much better |
| Min LR | 3e-5 | 1e-5 | Allow finer convergence |
| Warmup ratio | 0.1 | 0.05 | Shorter warmup, more training at peak LR |

### Expected Outcome After Cleaning

| Metric | Before | After (Est.) |
|--------|--------|--------------|
| Total Q&A rows | 96,245 | ~70,000 |
| Templated examples | ~30% | <5% |
| Truncated garbage | 5.5% | 0% |
| Book data ratio | 3.8% | ~5.5% |
| Boilerplate | 83.4% | 0% |

### Priority Order

1. **Filter >1024 tokens** (easy, clear cut)
2. **Deduplicate templates** (harder, need to identify patterns)
3. **Retrain with 10 epochs**
4. **Evaluate and iterate**
