"""
Cache contexts from database for HPC testing.
Run this LOCALLY before uploading to HPC.

Run: python cache_contexts.py
Output: cached_contexts.json
"""

import json
from retrieval import Retriever


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


def build_context(results: list) -> str:
    parts = []
    for r in results[:3]:
        q = r['question']
        a = r['answer'][:300]
        parts.append(f"Q: {q} A: {a}")
    return " ".join(parts)


def main():
    print("=" * 60)
    print("CACHE CONTEXTS FOR HPC")
    print("=" * 60)
    
    retriever = Retriever()
    
    cached = []
    for i, tc in enumerate(TEST_CASES, 1):
        print(f"[{i}/20] {tc['question'][:50]}...")
        
        retrieved = retriever.search(tc["question"], top_k=5)
        context = build_context(retrieved)
        prompt = f"Context: {context} Question: {tc['question']} Answer:"
        
        cached.append({
            "question": tc["question"],
            "keywords": tc["keywords"],
            "prompt": prompt
        })
    
    output_file = "cached_contexts.json"
    with open(output_file, "w") as f:
        json.dump(cached, f, indent=2)
    
    print()
    print(f"Saved {len(cached)} contexts to {output_file}")
    print()
    print("Next steps:")
    print("  1. Upload cached_contexts.json to HPC")
    print("  2. Upload param_battery_hpc.py to HPC")
    print("  3. Run: sbatch run_param_test.sh")


if __name__ == "__main__":
    main()
