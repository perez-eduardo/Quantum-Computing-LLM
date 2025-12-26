"""
Test different generation parameters to find optimal settings.
Focuses on the 3 failing questions.

Run: python tune_generation.py
"""

import time
from retrieval import Retriever
from inference import QuantumInference


# The 3 failing questions
TEST_CASES = [
    {
        "question": "What is a quantum gate?",
        "keywords": ["gate", "operation", "qubit", "unitary", "transform"]
    },
    {
        "question": "What is the Hadamard gate?",
        "keywords": ["hadamard", "superposition", "gate", "equal", "probability"]
    },
    {
        "question": "What is a quantum circuit?",
        "keywords": ["circuit", "gate", "qubit", "operation", "sequence"]
    },
]

# Parameters to test
TEMPERATURES = [0.3, 0.5, 0.7]
TOP_K_VALUES = [20, 50]


def check_keywords(answer: str, keywords: list) -> tuple:
    answer_lower = answer.lower()
    found = [kw for kw in keywords if kw.lower() in answer_lower]
    return found, len(found) / len(keywords)


def build_context(results: list) -> str:
    parts = []
    for r in results[:3]:
        q = r['question']
        a = r['answer'][:300]
        parts.append(f"Q: {q} A: {a}")
    return " ".join(parts)


def test_params(inferencer, retriever, temperature, top_k):
    """Test a specific parameter combination."""
    results = []
    
    for tc in TEST_CASES:
        question = tc["question"]
        keywords = tc["keywords"]
        
        # Retrieve
        retrieved = retriever.search(question, top_k=5)
        context = build_context(retrieved)
        prompt = f"Context: {context} Question: {question} Answer:"
        
        # Generate
        generated = inferencer.generate(
            prompt,
            max_new_tokens=100,
            temperature=temperature,
            top_k=top_k
        )
        answer = inferencer.extract_answer(generated)
        
        # Score
        found, score = check_keywords(answer, keywords)
        
        results.append({
            "question": question,
            "answer": answer[:150],
            "found": found,
            "score": score
        })
    
    return results


def main():
    print("=" * 70)
    print("TUNE GENERATION PARAMETERS")
    print("=" * 70)
    print(f"Testing {len(TEST_CASES)} failing questions")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Top-k values: {TOP_K_VALUES}")
    
    # Initialize
    print("\nInitializing...")
    retriever = Retriever()
    inferencer = QuantumInference()
    
    # Test each combination
    all_results = {}
    
    for temp in TEMPERATURES:
        for top_k in TOP_K_VALUES:
            combo = f"temp={temp}, top_k={top_k}"
            print(f"\n{'='*70}")
            print(f"Testing: {combo}")
            print("=" * 70)
            
            start = time.time()
            results = test_params(inferencer, retriever, temp, top_k)
            elapsed = time.time() - start
            
            avg_score = sum(r["score"] for r in results) / len(results)
            all_results[combo] = {
                "results": results,
                "avg_score": avg_score,
                "time": elapsed
            }
            
            for r in results:
                status = "PASS" if r["score"] >= 0.4 else "FAIL"
                print(f"\n  {status}: {r['question']}")
                print(f"    Keywords: {r['found']} ({r['score']:.0%})")
                print(f"    Answer: {r['answer'][:100]}...")
            
            print(f"\n  Avg score: {avg_score:.0%} | Time: {elapsed:.1f}s")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Sort by score
    ranked = sorted(all_results.items(), key=lambda x: x[1]["avg_score"], reverse=True)
    
    print("\nRanked by average score:")
    for combo, data in ranked:
        print(f"  {combo}: {data['avg_score']:.0%}")
    
    best = ranked[0]
    print(f"\n>>> BEST: {best[0]} with {best[1]['avg_score']:.0%} accuracy")


if __name__ == "__main__":
    main()
