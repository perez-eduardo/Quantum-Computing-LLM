"""
Debug script to inspect what's being sent to the model.
Shows retrieved context and full prompt.

Run: python debug_pipeline.py "What is a qubit?"
"""

import sys
from retrieval import Retriever
from inference import QuantumInference


def debug_query(question: str):
    print("=" * 70)
    print("DEBUG: RAG PIPELINE")
    print("=" * 70)
    
    # Step 1: Retrieve
    print(f"\n[1] QUESTION: {question}")
    print("=" * 70)
    
    retriever = Retriever()
    results = retriever.search(question, top_k=3)
    
    print(f"\n[2] RETRIEVED CONTEXT ({len(results)} results):")
    print("=" * 70)
    
    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (sim={r['similarity']:.4f}, source={r['source']}) ---")
        print(f"Q: {r['question'][:100]}...")
        print(f"A: {r['answer'][:200]}...")
    
    # Step 2: Build context
    context_parts = []
    for r in results:
        q = r['question']
        a = r['answer'][:300]  # Truncate
        context_parts.append(f"Q: {q} A: {a}")
    
    context = " ".join(context_parts)
    
    print(f"\n[3] BUILT CONTEXT ({len(context)} chars):")
    print("=" * 70)
    print(context[:500] + "..." if len(context) > 500 else context)
    
    # Step 3: Build prompt
    prompt = f"Context: {context} Question: {question} Answer:"
    
    print(f"\n[4] FULL PROMPT ({len(prompt)} chars):")
    print("=" * 70)
    print(prompt[:800] + "..." if len(prompt) > 800 else prompt)
    
    # Step 4: Generate
    print(f"\n[5] GENERATING (this takes ~60s on CPU)...")
    print("=" * 70)
    
    inferencer = QuantumInference()
    generated = inferencer.generate(prompt, max_new_tokens=100, temperature=0.7)
    
    print(f"\n[6] RAW OUTPUT:")
    print("=" * 70)
    print(generated)
    
    # Extract answer
    answer = inferencer.extract_answer(generated)
    
    print(f"\n[7] EXTRACTED ANSWER:")
    print("=" * 70)
    print(answer)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What is a qubit?"
    
    debug_query(question)
