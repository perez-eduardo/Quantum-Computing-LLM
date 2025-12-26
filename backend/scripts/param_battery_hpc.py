"""
Parameter Battery Test for HPC
Runs 180 generations using cached contexts (no DB/API calls).

Setup:
  1. Run cache_contexts.py locally first
  2. Upload cached_contexts.json to HPC
  3. sbatch run_param_test.sh

Run: python param_battery_hpc.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch

# Add scripts to path for model import
sys.path.insert(0, str(Path(__file__).parent))
from model import QuantumLLM


# 24 parameter combinations (4 temps × 6 top_k)
PARAM_COMBINATIONS = [
    # temp=0.1
    {"temperature": 0.1, "top_k": 10},
    {"temperature": 0.1, "top_k": 20},
    {"temperature": 0.1, "top_k": 30},
    {"temperature": 0.1, "top_k": 40},
    {"temperature": 0.1, "top_k": 50},
    {"temperature": 0.1, "top_k": 60},
    # temp=0.2
    {"temperature": 0.2, "top_k": 10},
    {"temperature": 0.2, "top_k": 20},
    {"temperature": 0.2, "top_k": 30},
    {"temperature": 0.2, "top_k": 40},
    {"temperature": 0.2, "top_k": 50},
    {"temperature": 0.2, "top_k": 60},
    # temp=0.3
    {"temperature": 0.3, "top_k": 10},
    {"temperature": 0.3, "top_k": 20},
    {"temperature": 0.3, "top_k": 30},
    {"temperature": 0.3, "top_k": 40},
    {"temperature": 0.3, "top_k": 50},  # baseline
    {"temperature": 0.3, "top_k": 60},
    # temp=0.4
    {"temperature": 0.4, "top_k": 10},
    {"temperature": 0.4, "top_k": 20},
    {"temperature": 0.4, "top_k": 30},
    {"temperature": 0.4, "top_k": 40},
    {"temperature": 0.4, "top_k": 50},
    {"temperature": 0.4, "top_k": 60},
]


class InferenceEngine:
    """Minimal inference engine for HPC."""
    
    def __init__(self, model_path, tokenizer_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Load model
        print(f"Loading model: {model_path}")
        self.model = QuantumLLM.load(str(model_path), self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded: {self.model.n_params:,} parameters")
        
        # Load tokenizer
        from tokenizers import Tokenizer
        print(f"Loading tokenizer: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"Tokenizer loaded: {self.tokenizer.get_vocab_size()} tokens")
    
    def generate(self, prompt, max_new_tokens=100, temperature=0.3, top_k=50):
        tokens = self.tokenizer.encode(prompt).ids
        x = torch.tensor([tokens], device=self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
        
        return self.tokenizer.decode(output[0].tolist())
    
    def extract_answer(self, generated_text):
        question_marker = "Question:"
        answer_marker = "Answer:"
        
        question_idx = generated_text.find(question_marker)
        
        if question_idx == -1:
            answer_idx = generated_text.find(answer_marker)
            if answer_idx == -1:
                return generated_text.strip()
            answer = generated_text[answer_idx + len(answer_marker):].strip()
        else:
            search_start = question_idx + len(question_marker)
            answer_idx = generated_text.find(answer_marker, search_start)
            
            if answer_idx == -1:
                return generated_text[search_start:].strip()
            
            answer = generated_text[answer_idx + len(answer_marker):].strip()
        
        stop_markers = ["Question:", "Q:", "Context:", "\n\n"]
        for stop in stop_markers:
            if stop in answer:
                answer = answer[:answer.find(stop)].strip()
        
        return answer


def check_keywords(answer, keywords):
    answer_lower = answer.lower()
    found = [kw for kw in keywords if kw.lower() in answer_lower]
    return found, len(found) / len(keywords) if keywords else 0


def run_param_test(engine, cached_contexts, temperature, top_k):
    results = []
    
    for i, item in enumerate(cached_contexts, 1):
        start = time.time()
        generated = engine.generate(
            item["prompt"],
            max_new_tokens=100,
            temperature=temperature,
            top_k=top_k
        )
        elapsed = time.time() - start
        
        answer = engine.extract_answer(generated)
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
        
        status = "PASS" if passed else "FAIL"
        print(f"\n[{i}/20] {item['question']}")
        print(f"  Status: {status} ({score:.0%}) | Time: {elapsed:.1f}s")
        print(f"  Answer: {answer[:200]}")
    
    return results


def main():
    print("=" * 70)
    print("PARAMETER BATTERY TEST (HPC)")
    print("=" * 70)
    
    # Paths for HPC: ~/hpc-share/quantum-llm/scripts/
    script_dir = Path(__file__).parent
    model_path = script_dir.parent / "model" / "final_model.pt"
    tokenizer_path = script_dir.parent / "tokenizer" / "tokenizer.json"
    contexts_path = script_dir / "cached_contexts.json"
    
    # Check files exist
    if not contexts_path.exists():
        print(f"ERROR: {contexts_path} not found")
        print("Run cache_contexts.py locally first!")
        return
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return
    
    if not tokenizer_path.exists():
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        return
    
    # Load cached contexts
    print(f"\nLoading cached contexts from {contexts_path}")
    with open(contexts_path) as f:
        cached_contexts = json.load(f)
    print(f"Loaded {len(cached_contexts)} contexts")
    
    print(f"\nParameter combinations: {len(PARAM_COMBINATIONS)}")
    print(f"Total generations: {len(cached_contexts) * len(PARAM_COMBINATIONS)}")
    print()
    
    # Initialize model
    engine = InferenceEngine(model_path, tokenizer_path)
    
    all_results = {}
    start_time = time.time()
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
        
        results = run_param_test(engine, cached_contexts, temp, top_k)
        
        passed = sum(1 for r in results if r["passed"])
        avg_score = sum(r["score"] for r in results) / len(results)
        avg_time = sum(r["time"] for r in results) / len(results)
        
        all_results[combo_name] = {
            "params": params,
            "results": results,
            "passed": passed,
            "total": len(results),
            "pass_rate": passed / len(results),
            "avg_score": avg_score,
            "avg_time": avg_time,
            "is_baseline": is_baseline
        }
        
        print()
        print("-" * 70)
        print(f"COMBO SUMMARY: {combo_name}")
        print(f"  Passed: {passed}/{len(results)} ({passed/len(results):.0%})")
        print(f"  Avg keyword score: {avg_score:.1%}")
        print(f"  Avg response time: {avg_time:.1f}s")
        
        elapsed = time.time() - start_time
        remaining = len(PARAM_COMBINATIONS) - combo_count
        eta = (elapsed / combo_count) * remaining
        print(f"  Elapsed: {elapsed/60:.1f} min | ETA: {eta/60:.1f} min")
    
    total_elapsed = time.time() - start_time
    
    # Final summary
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print()
    print(f"{'Parameters':<20} {'Pass Rate':<12} {'Keyword Score':<15} {'Avg Time':<10}")
    print("-" * 60)
    
    ranked = sorted(
        all_results.items(),
        key=lambda x: (x[1]["pass_rate"], x[1]["avg_score"]),
        reverse=True
    )
    
    for combo_name, data in ranked:
        marker = " *" if data.get("is_baseline") else ""
        print(f"{combo_name:<20} {data['pass_rate']:<12.0%} {data['avg_score']:<15.1%} {data['avg_time']:<10.1f}s{marker}")
    
    print("\n* = baseline (temp=0.3, top_k=50)")
    
    best_name, best_data = ranked[0]
    baseline_data = all_results.get("temp=0.3, top_k=50", {})
    
    print()
    print("=" * 70)
    print(f"BEST PARAMS: {best_name}")
    print(f"  Pass rate: {best_data['pass_rate']:.0%}")
    print(f"  Keyword score: {best_data['avg_score']:.1%}")
    print(f"  Avg time: {best_data['avg_time']:.1f}s")
    print("=" * 70)
    
    if baseline_data and best_name != "temp=0.3, top_k=50":
        print()
        print("COMPARISON TO BASELINE:")
        print(f"  Pass rate:     {baseline_data['pass_rate']:.0%} -> {best_data['pass_rate']:.0%}")
        print(f"  Keyword score: {baseline_data['avg_score']:.1%} -> {best_data['avg_score']:.1%}")
    
    failures = [r for r in best_data["results"] if not r["passed"]]
    if failures:
        print(f"\nFailures with best params ({len(failures)}):")
        for r in failures:
            print(f"  - {r['question']}")
            print(f"    Score: {r['score']:.0%}")
            print(f"    Answer: {r['answer'][:150]}")
    else:
        print("\nNo failures with best params!")
    
    print()
    print(f"Total generations: {len(cached_contexts) * len(PARAM_COMBINATIONS)}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    
    # Save results
    output_file = f"param_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    json_results = {
        "summary": {
            "best_params": best_name,
            "best_pass_rate": best_data["pass_rate"],
            "best_keyword_score": best_data["avg_score"],
            "total_time_minutes": total_elapsed / 60
        },
        "rankings": [
            {"rank": i+1, "params": name, "pass_rate": data["pass_rate"], "keyword_score": data["avg_score"]}
            for i, (name, data) in enumerate(ranked)
        ],
        "details": {
            name: {
                "params": data["params"],
                "passed": data["passed"],
                "pass_rate": data["pass_rate"],
                "avg_score": data["avg_score"],
                "questions": [
                    {"question": r["question"], "answer": r["answer"], "score": r["score"], "passed": r["passed"]}
                    for r in data["results"]
                ]
            }
            for name, data in all_results.items()
        }
    }
    
    with open(output_file, "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
