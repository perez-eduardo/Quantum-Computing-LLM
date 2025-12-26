"""
Live Deployment Simulation Test
Full RAG pipeline for each query (Voyage API + Neon DB + Model)

Run: python verify_params.py
"""

import time
import json
from datetime import datetime
from retrieval import Retriever
from inference import QuantumInference


TEST_CASES = [
    {"question": "What is a qubit?", "keywords": ["qubit", "quantum", "bit", "state", "superposition"]},
    {"question": "What is quantum entanglement?", "keywords": ["entangle", "correlat", "particle", "measure", "state"]},
    {"question": "What is superposition?", "keywords": ["superposition", "multiple", "state", "simultaneous"]},
    {"question": "What is a quantum gate?", "keywords": ["gate", "operation", "qubit", "unitary", "transform"]},
    {"question": "What is Shor's algorithm?", "keywords": ["shor", "factor", "prime", "exponential", "cryptograph"]},
    {"question": "What is Grover's algorithm?", "keywords": ["grover", "search", "speedup", "database"]},
    {"question": "What is quantum error correction?", "keywords": ["error", "correct", "logical", "code"]},
    {"question": "What is the Hadamard gate?", "keywords": ["hadamard", "superposition", "gate", "equal", "probability"]},
    {"question": "What is quantum decoherence?", "keywords": ["decoherence", "environment", "loss", "coherence", "noise"]},
    {"question": "What is a quantum circuit?", "keywords": ["circuit", "gate", "qubit", "operation", "sequence"]},
    {"question": "What is quantum computing?", "keywords": ["quantum", "comput", "qubit", "classical"]},
    {"question": "What is the Bloch sphere?", "keywords": ["bloch", "sphere", "qubit", "state", "visual"]},
    {"question": "What is quantum teleportation?", "keywords": ["teleport", "entangle", "state", "transfer", "classical"]},
    {"question": "What is a CNOT gate?", "keywords": ["cnot", "control", "target", "gate", "qubit"]},
    {"question": "What is quantum supremacy?", "keywords": ["supremacy", "advantage", "classical", "faster"]},
    {"question": "What is a quantum register?", "keywords": ["register", "qubit", "multiple", "state"]},
    {"question": "What is measurement in quantum computing?", "keywords": ["measure", "collapse", "state", "outcome", "probabilit"]},
    {"question": "What is quantum interference?", "keywords": ["interfere", "amplitude", "wave", "cancel", "probabilit"]},
    {"question": "What is a universal quantum gate set?", "keywords": ["universal", "gate", "set", "comput", "any"]},
    {"question": "What is quantum parallelism?", "keywords": ["parallel", "superposition", "simultaneous", "comput"]},
]

PARAM_COMBINATIONS = [
    {"temperature": 0.2, "top_k": 30},
    {"temperature": 0.4, "top_k": 20},
]


def check_keywords(answer, keywords):
    answer_lower = answer.lower()
    found = [kw for kw in keywords if kw.lower() in answer_lower]
    return found, len(found) / len(keywords) if keywords else 0


def build_context(results):
    parts = []
    for r in results[:3]:
        parts.append(f"Q: {r['question']} A: {r['answer'][:300]}")
    return " ".join(parts)


def run_full_pipeline(retriever, inferencer, question, temperature, top_k):
    """
    Simulate live deployment: full RAG pipeline for single query.
    1. Voyage API embeds query
    2. Neon DB retrieves context
    3. Model generates answer
    """
    total_start = time.time()
    
    # Step 1: Retrieval (Voyage + Neon)
    retrieve_start = time.time()
    retrieved = retriever.search(question, top_k=5)
    retrieve_time = time.time() - retrieve_start
    
    # Step 2: Build prompt
    context = build_context(retrieved)
    prompt = f"Context: {context} Question: {question} Answer:"
    
    # Step 3: Generate (Model)
    gen_start = time.time()
    generated = inferencer.generate(
        prompt,
        max_new_tokens=100,
        temperature=temperature,
        top_k=top_k
    )
    gen_time = time.time() - gen_start
    
    # Step 4: Extract answer
    answer = inferencer.extract_answer(generated)
    
    total_time = time.time() - total_start
    
    return {
        "answer": answer,
        "retrieve_time": retrieve_time,
        "gen_time": gen_time,
        "total_time": total_time
    }


def main():
    print("=" * 70)
    print("LIVE DEPLOYMENT SIMULATION TEST")
    print("=" * 70)
    print("Full RAG pipeline: Voyage API -> Neon DB -> Custom Model")
    print()
    print(f"Questions: {len(TEST_CASES)}")
    print(f"Combinations: {len(PARAM_COMBINATIONS)}")
    print(f"Total tests: {len(TEST_CASES) * len(PARAM_COMBINATIONS)}")
    print()

    print("Initializing...")
    retriever = Retriever()
    inferencer = QuantumInference()
    print()

    all_results = {}
    start_time = time.time()

    for combo_idx, params in enumerate(PARAM_COMBINATIONS, 1):
        temp = params["temperature"]
        top_k = params["top_k"]
        combo_name = f"temp={temp}, top_k={top_k}"

        print()
        print("=" * 70)
        print(f"[{combo_idx}/{len(PARAM_COMBINATIONS)}] TESTING: {combo_name}")
        print("=" * 70)

        results = []
        for i, tc in enumerate(TEST_CASES, 1):
            # Full pipeline simulation
            output = run_full_pipeline(
                retriever, inferencer,
                tc["question"],
                temp, top_k
            )
            
            found, score = check_keywords(output["answer"], tc["keywords"])
            passed = score >= 0.4

            results.append({
                "question": tc["question"],
                "answer": output["answer"],
                "score": score,
                "passed": passed,
                "retrieve_time": output["retrieve_time"],
                "gen_time": output["gen_time"],
                "total_time": output["total_time"]
            })

            status = "PASS" if passed else "FAIL"
            print(f"\n[{i}/20] {tc['question']}")
            print(f"  Status: {status} ({score:.0%})")
            print(f"  Time: {output['total_time']:.1f}s (retrieve: {output['retrieve_time']:.2f}s, gen: {output['gen_time']:.1f}s)")
            print(f"  Answer: {output['answer'][:200]}")

        passed_count = sum(1 for r in results if r["passed"])
        avg_score = sum(r["score"] for r in results) / len(results)
        avg_retrieve = sum(r["retrieve_time"] for r in results) / len(results)
        avg_gen = sum(r["gen_time"] for r in results) / len(results)
        avg_total = sum(r["total_time"] for r in results) / len(results)

        all_results[combo_name] = {
            "params": params,
            "passed": passed_count,
            "total": len(results),
            "pass_rate": passed_count / len(results),
            "avg_score": avg_score,
            "avg_retrieve_time": avg_retrieve,
            "avg_gen_time": avg_gen,
            "avg_total_time": avg_total,
            "results": results
        }

        print()
        print("-" * 70)
        print(f"SUMMARY: {combo_name}")
        print(f"  Passed: {passed_count}/{len(results)} ({passed_count/len(results):.0%})")
        print(f"  Avg keyword score: {avg_score:.1%}")
        print(f"  Avg retrieval time: {avg_retrieve:.2f}s")
        print(f"  Avg generation time: {avg_gen:.1f}s")
        print(f"  Avg total time: {avg_total:.1f}s")

    total_time = time.time() - start_time

    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Parameters':<20} {'Pass Rate':<12} {'Keyword':<10} {'Retrieve':<10} {'Generate':<10} {'Total':<10}")
    print("-" * 72)

    for combo_name, data in all_results.items():
        print(f"{combo_name:<20} {data['pass_rate']:<12.0%} {data['avg_score']:<10.1%} {data['avg_retrieve_time']:<10.2f}s {data['avg_gen_time']:<10.1f}s {data['avg_total_time']:<10.1f}s")

    print()
    print(f"Total time: {total_time/60:.1f} minutes")

    output_file = f"verify_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump({
            k: {key: val for key, val in v.items() if key != "results"} 
            for k, v in all_results.items()
        }, f, indent=2)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
