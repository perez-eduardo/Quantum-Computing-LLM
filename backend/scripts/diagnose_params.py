"""
Diagnostic: Test all 10 questions with temp=0.3, top_k=50
Compare to baseline (temp=0.7, top_k=50)

Run: python diagnose_params.py
"""

import time
from retrieval import Retriever
from inference import QuantumInference


TEST_CASES = [
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


def run_test(inferencer, retriever, temperature, top_k):
    """Run all 10 tests with specific params."""
    results = []
    total_time = 0
    
    for i, tc in enumerate(TEST_CASES):
        question = tc["question"]
        keywords = tc["keywords"]
        
        print(f"\n[{i+1}/10] {question}")
        
        # Retrieve
        retrieved = retriever.search(question, top_k=50)
        context = build_context(retrieved)
        prompt = f"Context: {context} Question: {question} Answer:"
        
        # Generate with explicit params
        start = time.time()
        generated = inferencer.generate(
            prompt,
            max_new_tokens=100,
            temperature=temperature,
            top_k=top_k
        )
        elapsed = time.time() - start
        total_time += elapsed
        
        answer = inferencer.extract_answer(generated)
        
        # Score
        found, score = check_keywords(answer, keywords)
        passed = score >= 0.4
        
        results.append({
            "question": question,
            "answer": answer[:100],
            "found": found,
            "score": score,
            "passed": passed,
            "time": elapsed
        })
        
        status = "PASS" if passed else "FAIL"
        print(f"  Status: {status} ({score:.0%})")
        print(f"  Keywords: {found}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Answer: {answer[:80]}...")
    
    return results, total_time


def main():
    print("=" * 70)
    print("DIAGNOSE GENERATION PARAMS")
    print("=" * 70)
    print("Testing all 10 questions with temp=0.3, top_k=50")
    print("Baseline was: temp=0.7, top_k=50 (70% pass, 43.8% keyword score)")
    
    # Initialize
    print("\nInitializing...")
    retriever = Retriever()
    inferencer = QuantumInference()
    
    # Test with new params
    print("\n" + "=" * 70)
    print("TESTING: temp=0.3, top_k=50")
    print("=" * 70)
    
    results, total_time = run_test(inferencer, retriever, temperature=0.3, top_k=50)
    
    # Summary
    passed = sum(1 for r in results if r["passed"])
    avg_score = sum(r["score"] for r in results) / len(results)
    avg_time = total_time / len(results)
    
    print("\n" + "=" * 70)
    print("RESULTS: temp=0.3, top_k=50")
    print("=" * 70)
    print(f"Passed: {passed}/10 ({passed*10}%)")
    print(f"Average keyword score: {avg_score:.1%}")
    print(f"Average response time: {avg_time:.1f}s")
    print(f"Total time: {total_time:.1f}s")
    
    # Compare to baseline
    print("\n" + "=" * 70)
    print("COMPARISON TO BASELINE")
    print("=" * 70)
    print(f"                    Baseline    New Params")
    print(f"Pass rate:          70%         {passed*10}%")
    print(f"Keyword score:      43.8%       {avg_score:.1%}")
    print(f"Params:             0.7/50      0.3/20")
    
    # Show failures
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for r in failures:
            print(f"  - {r['question']}")
            print(f"    Found: {r['found']} ({r['score']:.0%})")
    
    # Verdict
    print("\n" + "=" * 70)
    if passed >= 7 and avg_score >= 0.438:
        print("VERDICT: New params are SAME OR BETTER. Safe to update defaults.")
    elif passed > 7 or avg_score > 0.438:
        print("VERDICT: New params show IMPROVEMENT. Recommend updating defaults.")
    else:
        print("VERDICT: New params are WORSE. Keep current defaults.")
    print("=" * 70)


if __name__ == "__main__":
    main()
