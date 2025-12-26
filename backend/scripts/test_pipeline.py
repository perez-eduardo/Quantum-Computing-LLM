"""
Test script for Quantum Computing RAG Pipeline.
Runs comprehensive tests and reports results.

Usage:
    python test_pipeline.py
    python test_pipeline.py --interactive
"""

import argparse
import time
from pipeline import QuantumRAGPipeline


# Test cases with expected keywords
TEST_CASES = [
    {
        "question": "What is a qubit?",
        "keywords": ["quantum", "bit", "superposition", "state", "0", "1"]
    },
    {
        "question": "What is quantum entanglement?",
        "keywords": ["correlat", "particle", "measure", "state", "non-local"]
    },
    {
        "question": "What is superposition?",
        "keywords": ["multiple", "state", "simultaneous", "probability"]
    },
    {
        "question": "What is a quantum gate?",
        "keywords": ["operation", "qubit", "unitary", "transform"]
    },
    {
        "question": "What is Shor's algorithm?",
        "keywords": ["factor", "prime", "exponential", "cryptograph"]
    },
    {
        "question": "What is Grover's algorithm?",
        "keywords": ["search", "speedup", "quadratic", "database"]
    },
    {
        "question": "What is quantum error correction?",
        "keywords": ["error", "correct", "noise", "redundan", "logical"]
    },
    {
        "question": "What is the Hadamard gate?",
        "keywords": ["superposition", "gate", "equal", "probability"]
    },
    {
        "question": "What is quantum decoherence?",
        "keywords": ["environment", "noise", "loss", "coherence"]
    },
    {
        "question": "What is a quantum circuit?",
        "keywords": ["gate", "qubit", "operation", "sequence"]
    }
]


def check_keywords(answer: str, keywords: list) -> tuple:
    """Check how many keywords appear in the answer."""
    answer_lower = answer.lower()
    found = []
    missing = []
    
    for kw in keywords:
        if kw.lower() in answer_lower:
            found.append(kw)
        else:
            missing.append(kw)
    
    return found, missing


def run_tests(pipeline: QuantumRAGPipeline):
    """Run all test cases and report results."""
    print("\n" + "=" * 60)
    print("RUNNING TESTS")
    print("=" * 60)
    
    results = []
    total_time = 0
    
    for i, tc in enumerate(TEST_CASES, 1):
        question = tc["question"]
        keywords = tc["keywords"]
        
        print(f"\n[{i}/{len(TEST_CASES)}] {question}")
        
        start = time.time()
        response = pipeline.query(question)
        elapsed = time.time() - start
        total_time += elapsed
        
        answer = response["answer"]
        found, missing = check_keywords(answer, keywords)
        score = len(found) / len(keywords)
        
        status = "PASS" if score >= 0.3 else "FAIL"
        
        results.append({
            "question": question,
            "answer": answer,
            "keywords_found": found,
            "keywords_missing": missing,
            "score": score,
            "status": status,
            "time": elapsed,
            "suggestions": response["suggested_questions"]
        })
        
        print(f"  Status: {status} ({score:.0%})")
        print(f"  Keywords: {found}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Answer: {answer[:150]}...")
    
    # Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    avg_score = sum(r["score"] for r in results) / len(results)
    avg_time = total_time / len(results)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{len(TEST_CASES)} ({100*passed/len(TEST_CASES):.0f}%)")
    print(f"Average keyword score: {avg_score:.1%}")
    print(f"Average response time: {avg_time:.2f}s")
    print(f"Total time: {total_time:.1f}s")
    
    # Show failures
    failures = [r for r in results if r["status"] == "FAIL"]
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f['question']}")
            print(f"    Missing: {f['keywords_missing']}")
    
    return results


def interactive_mode(pipeline: QuantumRAGPipeline):
    """Interactive Q&A mode."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Type your questions (or 'quit' to exit)")
    
    while True:
        print("\n" + "-" * 40)
        question = input("You: ").strip()
        
        if not question:
            continue
        
        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        start = time.time()
        response = pipeline.query(question)
        elapsed = time.time() - start
        
        print(f"\nAssistant: {response['answer']}")
        print(f"\n[{elapsed:.2f}s]")
        
        if response["suggested_questions"]:
            print("\nYou might also ask:")
            for i, sq in enumerate(response["suggested_questions"], 1):
                print(f"  {i}. {sq}")


def main():
    parser = argparse.ArgumentParser(description="Test Quantum RAG Pipeline")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer JSON")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("QUANTUM RAG PIPELINE TEST")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = QuantumRAGPipeline(
        model_path=args.model,
        tokenizer_path=args.tokenizer
    )
    
    if args.interactive:
        interactive_mode(pipeline)
    else:
        run_tests(pipeline)


if __name__ == "__main__":
    main()
