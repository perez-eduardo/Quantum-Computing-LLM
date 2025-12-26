"""
Comprehensive Parameter Tuning Test (Optimized)
- Fetches 20 contexts ONCE from database
- Runs 9 param combinations locally (180 model generations)

Run: python param_battery_test.py
"""

import time
import json
from datetime import datetime
from retrieval import Retriever
from inference import QuantumInference


# 20 test questions
TEST_CASES = [
    # Original 10
    {
        "question": "What is a qubit?",
        "keywords": ["qubit", "quantum", "bit", "state", "superposition"]
    },
    {
        "question": "What is quantum entanglement?",
        "keywords": ["entangle", "correlat", "particle", "measure", "state"]
    },
    {
        "question": "What is superposition?",
        "keywords": ["superposition", "multiple", "state", "simultaneous"]
    },
    {
        "question": "What is a quantum gate?",
        "keywords": ["gate", "operation", "qubit", "unitary", "transform"]
    },
    {
        "question": "What is Shor's algorithm?",
        "keywords": ["shor", "factor", "prime", "exponential", "cryptograph"]
    },
    {
        "question": "What is Grover's algorithm?",
        "keywords": ["grover", "search", "speedup", "database"]
    },
    {
        "question": "What is quantum error correction?",
        "keywords": ["error", "correct", "logical", "code"]
    },
    {
        "question": "What is the Hadamard gate?",
        "keywords": ["hadamard", "superposition", "gate", "equal", "probability"]
    },
    {
        "question": "What is quantum decoherence?",
        "keywords": ["decoherence", "environment", "loss", "coherence", "noise"]
    },
    {
        "question": "What is a quantum circuit?",
        "keywords": ["circuit", "gate", "qubit", "operation", "sequence"]
    },
    # 10 new questions
    {
        "question": "What is quantum computing?",
        "keywords": ["quantum", "comput", "qubit", "classical"]
    },
    {
        "question": "What is the Bloch sphere?",
        "keywords": ["bloch", "sphere", "qubit", "state", "visual"]
    },
    {
        "question": "What is quantum teleportation?",
        "keywords": ["teleport", "entangle", "state", "transfer", "classical"]
    },
    {
        "question": "What is a CNOT gate?",
        "keywords": ["cnot", "control", "target", "gate", "qubit"]
    },
    {
        "question": "What is quantum supremacy?",
        "keywords": ["supremacy", "advantage", "classical", "faster"]
    },
    {
        "question": "What is a quantum register?",
        "keywords": ["register", "qubit", "multiple", "state"]
    },
    {
        "question": "What is measurement in quantum computing?",
        "keywords": ["measure", "collapse", "state", "outcome", "probabilit"]
    },
    {
        "question": "What is quantum interference?",
        "keywords": ["interfere", "amplitude", "wave", "cancel", "probabilit"]
    },
    {
        "question": "What is a universal quantum gate set?",
        "keywords": ["universal", "gate", "set", "comput", "any"]
    },
    {
        "question": "What is quantum parallelism?",
        "keywords": ["parallel", "superposition", "simultaneous", "comput"]
    },
]

# 9 parameter combinations
PARAM_COMBINATIONS = [
    {"temperature": 0.2, "top_k": 40},
    {"temperature": 0.2, "top_k": 50},
    {"temperature": 0.2, "top_k": 60},
    {"temperature": 0.3, "top_k": 40},
    {"temperature": 0.3, "top_k": 50},  # baseline
    {"temperature": 0.3, "top_k": 60},
    {"temperature": 0.4, "top_k": 40},
    {"temperature": 0.4, "top_k": 50},
    {"temperature": 0.4, "top_k": 60},
]


def check_keywords(answer: str, keywords: list) -> tuple:
    answer_lower = answer.lower()
    found = [kw for kw in keywords if kw.lower() in answer_lower]
    return found, len(found) / len(keywords) if keywords else 0


def build_context(results: list) -> str:
    parts = []
    for r in results[:3]:
        q = r['question']
        a = r['answer'][:300]
        parts.append(f"Q: {q} A: {a}")
    return " ".join(parts)


def fetch_all_contexts(retriever):
    """Fetch contexts for all 20 questions ONCE."""
    print("Fetching contexts from database (20 queries)...")
    
    cached = []
    for i, tc in enumerate(TEST_CASES, 1):
        retrieved = retriever.search(tc["question"], top_k=5)
        context = build_context(retrieved)
        prompt = f"Context: {context} Question: {tc['question']} Answer:"
        
        cached.append({
            "question": tc["question"],
            "keywords": tc["keywords"],
            "prompt": prompt
        })
        print(f"  [{i}/20] {tc['question'][:50]}...")
    
    print("Done. Contexts cached.\n")
    return cached


def run_generation(inferencer, prompt, temperature, top_k):
    """Run model generation with specific params."""
    start = time.time()
    generated = inferencer.generate(
        prompt,
        max_new_tokens=100,
        temperature=temperature,
        top_k=top_k
    )
    elapsed = time.time() - start
    answer = inferencer.extract_answer(generated)
    return answer, elapsed


def run_param_test(inferencer, cached_contexts, temperature, top_k):
    """Run all 20 questions with specific params using cached contexts."""
    results = []
    
    for i, item in enumerate(cached_contexts, 1):
        answer, elapsed = run_generation(
            inferencer, item["prompt"], temperature, top_k
        )
        
        found, score = check_keywords(answer, item["keywords"])
        passed = score >= 0.4
        
        result = {
            "question": item["question"],
            "answer": answer,
            "found": found,
            "score": score,
            "passed": passed,
            "time": elapsed
        }
        results.append(result)
        
        # Show output
        status = "PASS" if passed else "FAIL"
        print(f"\n[{i}/20] {item['question']}")
        print(f"  Status: {status} ({score:.0%}) | Time: {elapsed:.1f}s")
        print(f"  Answer: {answer[:200]}")
    
    return results


def main():
    print("=" * 70)
    print("COMPREHENSIVE PARAMETER TUNING TEST (OPTIMIZED)")
    print("=" * 70)
    print(f"Questions: {len(TEST_CASES)}")
    print(f"Parameter combinations: {len(PARAM_COMBINATIONS)}")
    print(f"Total generations: {len(TEST_CASES) * len(PARAM_COMBINATIONS)}")
    print(f"Database calls: 20 (cached)")
    print(f"Estimated time: ~{len(TEST_CASES) * len(PARAM_COMBINATIONS) * 35 // 60} minutes")
    print()
    
    # Initialize
    print("Initializing...")
    retriever = Retriever()
    inferencer = QuantumInference()
    print()
    
    # Fetch all contexts ONCE
    cached_contexts = fetch_all_contexts(retriever)
    
    # Store all results
    all_results = {}
    
    start_time = time.time()
    test_count = 0
    combo_count = 0
    
    for params in PARAM_COMBINATIONS:
        temp = params["temperature"]
        top_k = params["top_k"]
        combo_name = f"temp={temp}, top_k={top_k}"
        combo_count += 1
        
        is_baseline = (temp == 0.3 and top_k == 50)
        baseline_marker = " [BASELINE]" if is_baseline else ""
        
        print()
        print("=" * 70)
        print(f"[{combo_count}/{len(PARAM_COMBINATIONS)}] TESTING: {combo_name}{baseline_marker}")
        print("=" * 70)
        
        results = run_param_test(inferencer, cached_contexts, temp, top_k)
        test_count += len(results)
        
        # Calculate stats
        passed = sum(1 for r in results if r["passed"])
        avg_score = sum(r["score"] for r in results) / len(results)
        avg_time = sum(r["time"] for r in results) / len(results)
        total_time = sum(r["time"] for r in results)
        
        all_results[combo_name] = {
            "params": params,
            "results": results,
            "passed": passed,
            "total": len(results),
            "pass_rate": passed / len(results),
            "avg_score": avg_score,
            "avg_time": avg_time,
            "total_time": total_time,
            "is_baseline": is_baseline
        }
        
        print()
        print("-" * 70)
        print(f"COMBO SUMMARY: {combo_name}")
        print(f"  Passed: {passed}/{len(results)} ({passed/len(results):.0%})")
        print(f"  Avg keyword score: {avg_score:.1%}")
        print(f"  Avg response time: {avg_time:.1f}s")
        
        # Progress
        elapsed = time.time() - start_time
        remaining_combos = len(PARAM_COMBINATIONS) - combo_count
        if combo_count > 0:
            eta = (elapsed / combo_count) * remaining_combos
            print(f"  Elapsed: {elapsed/60:.1f} min | ETA: {eta/60:.1f} min remaining")
    
    total_elapsed = time.time() - start_time
    
    # Final summary
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print()
    print(f"{'Parameters':<20} {'Pass Rate':<12} {'Keyword Score':<15} {'Avg Time':<10}")
    print("-" * 60)
    
    # Sort by pass rate, then by keyword score
    ranked = sorted(
        all_results.items(),
        key=lambda x: (x[1]["pass_rate"], x[1]["avg_score"]),
        reverse=True
    )
    
    for combo_name, data in ranked:
        marker = " *" if data.get("is_baseline") else ""
        print(f"{combo_name:<20} {data['pass_rate']:<12.0%} {data['avg_score']:<15.1%} {data['avg_time']:<10.1f}s{marker}")
    
    print("\n* = baseline (temp=0.3, top_k=50)")
    
    # Best params
    best_name, best_data = ranked[0]
    baseline_data = all_results.get("temp=0.3, top_k=50", {})
    
    print()
    print("=" * 70)
    print(f"BEST PARAMS: {best_name}")
    print(f"  Pass rate: {best_data['pass_rate']:.0%}")
    print(f"  Keyword score: {best_data['avg_score']:.1%}")
    print(f"  Avg time: {best_data['avg_time']:.1f}s")
    print("=" * 70)
    
    # Compare to baseline
    if baseline_data and best_name != "temp=0.3, top_k=50":
        print()
        print("COMPARISON TO BASELINE (temp=0.3, top_k=50):")
        print(f"  Pass rate:     {baseline_data['pass_rate']:.0%} -> {best_data['pass_rate']:.0%}")
        print(f"  Keyword score: {baseline_data['avg_score']:.1%} -> {best_data['avg_score']:.1%}")
        print(f"  Avg time:      {baseline_data['avg_time']:.1f}s -> {best_data['avg_time']:.1f}s")
    
    # Show failures for best params
    failures = [r for r in best_data["results"] if not r["passed"]]
    if failures:
        print(f"\nFailures with best params ({len(failures)}):")
        for r in failures:
            print(f"  - {r['question']}")
            print(f"    Score: {r['score']:.0%}")
            print(f"    Answer: {r['answer'][:150]}")
    else:
        print("\nNo failures with best params!")
    
    # Total stats
    print()
    print(f"Total generations: {test_count}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Avg time per generation: {total_elapsed/test_count:.1f}s")
    
    # Save results to JSON
    output_file = f"param_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    json_results = {
        "summary": {
            "total_tests": test_count,
            "total_time_minutes": total_elapsed / 60,
            "best_params": best_name,
            "best_pass_rate": best_data["pass_rate"],
            "best_keyword_score": best_data["avg_score"]
        },
        "rankings": [
            {
                "rank": i + 1,
                "params": name,
                "pass_rate": data["pass_rate"],
                "keyword_score": data["avg_score"],
                "avg_time": data["avg_time"]
            }
            for i, (name, data) in enumerate(ranked)
        ],
        "details": {}
    }
    
    for combo_name, data in all_results.items():
        json_results["details"][combo_name] = {
            "params": data["params"],
            "passed": data["passed"],
            "total": data["total"],
            "pass_rate": data["pass_rate"],
            "avg_score": data["avg_score"],
            "avg_time": data["avg_time"],
            "questions": [
                {
                    "question": r["question"],
                    "answer": r["answer"],
                    "score": r["score"],
                    "passed": r["passed"],
                    "time": r["time"]
                }
                for r in data["results"]
            ]
        }
    
    with open(output_file, "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
