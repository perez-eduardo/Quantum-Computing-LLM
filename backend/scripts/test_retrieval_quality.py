"""
Test retrieval quality: Do retrieved chunks actually contain the answer?

Run from project root: python backend/scripts/test_retrieval_quality.py
"""

import os
from dotenv import load_dotenv
import voyageai
import psycopg2

load_dotenv()

EMBEDDING_MODEL = "voyage-3.5-lite"

# 100 questions with expected keywords that should appear in a correct answer
# Format: (question, [list of expected keywords/phrases that indicate relevance])
TEST_CASES = [
    # Basic concepts (1-20)
    ("What is a qubit?", ["quantum bit", "superposition", "0 and 1", "|0⟩", "|1⟩", "two-level"]),
    ("What is superposition?", ["multiple states", "simultaneously", "linear combination", "probability"]),
    ("What is quantum entanglement?", ["correlated", "non-local", "Bell", "EPR", "measured"]),
    ("What is quantum interference?", ["amplitude", "constructive", "destructive", "cancel", "probability"]),
    ("What is a quantum gate?", ["unitary", "operation", "transform", "matrix", "reversible"]),
    ("What is the Bloch sphere?", ["representation", "sphere", "surface", "theta", "phi", "visualization"]),
    ("What is quantum measurement?", ["collapse", "eigenvalue", "probability", "observe", "outcome"]),
    ("What is the no-cloning theorem?", ["cannot copy", "impossible", "duplicate", "unknown state"]),
    ("What is quantum decoherence?", ["environment", "noise", "loss", "classical", "interaction"]),
    ("What is a quantum circuit?", ["gates", "wires", "sequence", "operations", "diagram"]),
    ("What is quantum parallelism?", ["simultaneous", "superposition", "exponential", "all inputs"]),
    ("What is a tensor product?", ["combine", "systems", "⊗", "Hilbert space", "composite"]),
    ("What is the computational basis?", ["|0⟩", "|1⟩", "basis states", "standard basis", "orthonormal"]),
    ("What is quantum phase?", ["complex", "amplitude", "angle", "relative", "global"]),
    ("What is a pure state?", ["single", "ket", "vector", "not mixed", "superposition"]),
    ("What is a mixed state?", ["density matrix", "ensemble", "classical mixture", "probabilistic"]),
    ("What is the Born rule?", ["probability", "amplitude squared", "measurement", "|ψ|²"]),
    ("What is a Hilbert space?", ["vector space", "inner product", "complete", "complex", "dimension"]),
    ("What is quantum tunneling?", ["barrier", "penetrate", "probability", "classically forbidden"]),
    ("What is a quantum oracle?", ["black box", "function", "query", "input-output"]),
    
    # Gates (21-40)
    ("What is the Hadamard gate?", ["H gate", "superposition", "1/√2", "equal probability"]),
    ("What is the Pauli-X gate?", ["NOT gate", "bit flip", "X gate", "0 to 1", "σx"]),
    ("What is the Pauli-Y gate?", ["Y gate", "rotation", "σy", "complex"]),
    ("What is the Pauli-Z gate?", ["phase flip", "Z gate", "σz", "|1⟩ to -|1⟩"]),
    ("What is the CNOT gate?", ["controlled", "NOT", "two-qubit", "target", "control", "entangle"]),
    ("What is the SWAP gate?", ["exchange", "swap", "two qubits", "interchange"]),
    ("What is the Toffoli gate?", ["CCNOT", "three-qubit", "universal", "reversible", "controlled-controlled"]),
    ("What is the T gate?", ["π/8", "phase", "π/4", "rotation"]),
    ("What is the S gate?", ["phase gate", "π/2", "√Z", "rotation"]),
    ("What is a controlled gate?", ["control qubit", "target", "conditional", "CU"]),
    ("What is the Fredkin gate?", ["controlled swap", "CSWAP", "three-qubit", "reversible"]),
    ("What are universal gates?", ["any operation", "complete set", "approximate", "H", "T", "CNOT"]),
    ("What is a rotation gate?", ["Rx", "Ry", "Rz", "angle", "axis", "continuous"]),
    ("What is the phase gate?", ["diagonal", "phase shift", "R", "rotation"]),
    ("How do you create entanglement with gates?", ["Hadamard", "CNOT", "Bell state", "two qubits"]),
    ("What is gate decomposition?", ["universal", "break down", "sequence", "approximate"]),
    ("What is gate fidelity?", ["accuracy", "error", "ideal", "closeness"]),
    ("What is a parametrized gate?", ["angle", "parameter", "variational", "θ"]),
    ("What is the CZ gate?", ["controlled-Z", "phase", "symmetric", "two-qubit"]),
    ("What is the iSWAP gate?", ["swap", "phase", "i", "fermionic"]),
    
    # Algorithms (41-60)
    ("What is Shor's algorithm?", ["factoring", "prime", "exponential", "RSA", "period finding"]),
    ("What is Grover's algorithm?", ["search", "unstructured", "quadratic", "√N", "amplitude amplification"]),
    ("What is the Deutsch algorithm?", ["oracle", "constant", "balanced", "single query"]),
    ("What is the Deutsch-Jozsa algorithm?", ["constant or balanced", "exponential speedup", "oracle"]),
    ("What is quantum phase estimation?", ["eigenvalue", "phase", "unitary", "precision"]),
    ("What is the quantum Fourier transform?", ["QFT", "Fourier", "superposition", "phase"]),
    ("What is amplitude amplification?", ["Grover", "increase probability", "rotation", "oracle"]),
    ("What is the Bernstein-Vazirani algorithm?", ["hidden string", "inner product", "one query"]),
    ("What is Simon's algorithm?", ["period", "hidden subgroup", "exponential speedup"]),
    ("What is the HHL algorithm?", ["linear systems", "matrix inversion", "Ax=b", "exponential"]),
    ("What is VQE?", ["variational", "eigensolver", "ansatz", "hybrid", "classical optimizer"]),
    ("What is QAOA?", ["optimization", "combinatorial", "approximate", "MaxCut", "variational"]),
    ("What is quantum walk?", ["random walk", "superposition", "graph", "speedup"]),
    ("What is adiabatic quantum computing?", ["ground state", "Hamiltonian", "slowly", "gap"]),
    ("What is quantum annealing?", ["optimization", "D-Wave", "thermal", "tunneling"]),
    ("What is quantum machine learning?", ["learning", "quantum data", "kernel", "classification"]),
    ("What is the swap test?", ["overlap", "inner product", "fidelity", "compare states"]),
    ("What is quantum counting?", ["Grover", "number of solutions", "phase estimation"]),
    ("What is the quantum approximate counting?", ["estimate", "solutions", "amplitude estimation"]),
    ("What is quantum simulation?", ["simulate", "Hamiltonian", "physical system", "exponential"]),
    
    # Error correction (61-75)
    ("What is quantum error correction?", ["errors", "redundancy", "logical qubit", "physical qubits"]),
    ("What is the bit flip code?", ["three qubits", "X error", "majority", "repetition"]),
    ("What is the phase flip code?", ["Z error", "Hadamard", "three qubits"]),
    ("What is the Shor code?", ["nine qubits", "bit flip", "phase flip", "concatenated"]),
    ("What is the Steane code?", ["seven qubits", "CSS", "fault-tolerant"]),
    ("What is a stabilizer code?", ["stabilizer", "Pauli", "group", "eigenstate"]),
    ("What is the surface code?", ["2D", "lattice", "threshold", "topological"]),
    ("What is fault-tolerant quantum computing?", ["errors", "threshold", "propagate", "logical"]),
    ("What is the threshold theorem?", ["error rate", "arbitrary", "below threshold", "scalable"]),
    ("What are Pauli errors?", ["X", "Y", "Z", "bit flip", "phase flip"]),
    ("What is a logical qubit?", ["encoded", "physical qubits", "protected", "error correction"]),
    ("What is syndrome measurement?", ["error", "detect", "ancilla", "without destroying"]),
    ("What is a CSS code?", ["Calderbank-Shor-Steane", "classical codes", "X and Z"]),
    ("What is quantum fault tolerance?", ["gates", "errors", "spread", "threshold"]),
    ("What is magic state distillation?", ["T gate", "fault-tolerant", "noisy", "purify"]),
    
    # Hardware (76-90)
    ("What is a superconducting qubit?", ["Josephson junction", "transmon", "microwave", "dilution"]),
    ("What is an ion trap qubit?", ["trapped ion", "laser", "electromagnetic", "long coherence"]),
    ("What is a photonic qubit?", ["photon", "polarization", "linear optics", "beam splitter"]),
    ("What is a topological qubit?", ["Majorana", "braiding", "protected", "anyons"]),
    ("What is quantum coherence time?", ["T1", "T2", "decoherence", "how long"]),
    ("What is T1 relaxation?", ["energy", "decay", "ground state", "amplitude damping"]),
    ("What is T2 dephasing?", ["phase", "coherence", "superposition", "T2*"]),
    ("What is a dilution refrigerator?", ["millikelvin", "cooling", "superconducting", "cryogenic"]),
    ("What is quantum volume?", ["metric", "benchmark", "depth", "IBM"]),
    ("What is NISQ?", ["noisy", "intermediate", "scale", "near-term"]),
    ("What is quantum supremacy?", ["advantage", "classical", "impossible", "speedup"]),
    ("What are neutral atom qubits?", ["atoms", "optical tweezers", "Rydberg", "array"]),
    ("What is a transmon qubit?", ["superconducting", "charge", "anharmonic", "capacitor"]),
    ("What is readout error?", ["measurement", "misidentify", "0 as 1", "fidelity"]),
    ("What is gate error rate?", ["fidelity", "imperfect", "noise", "two-qubit"]),
    
    # Math and notation (91-100)
    ("What is a density matrix?", ["ρ", "mixed state", "trace", "positive semidefinite"]),
    ("What is the trace operation?", ["sum", "diagonal", "Tr", "probability"]),
    ("What is the partial trace?", ["subsystem", "reduced", "trace out", "entanglement"]),
    ("What is operator notation?", ["bra", "ket", "Dirac", "⟨", "⟩"]),
    ("What is an eigenstate?", ["eigenvalue", "eigenvector", "measurement", "definite value"]),
    ("What is a unitary matrix?", ["U†U = I", "reversible", "norm preserving", "inverse"]),
    ("What is a Hermitian operator?", ["observable", "real eigenvalues", "self-adjoint", "H† = H"]),
    ("What is the commutator?", ["[A,B]", "AB - BA", "compatible", "uncertainty"]),
    ("What is fidelity?", ["similarity", "overlap", "closeness", "F(ρ,σ)"]),
    ("What is the Pauli group?", ["I", "X", "Y", "Z", "matrices", "multiplication"]),
]


class Retriever:
    def __init__(self):
        api_key = os.getenv("VOYAGE_API_KEY")
        db_url = os.getenv("DATABASE_URL")
        self.voyage_client = voyageai.Client(api_key=api_key)
        self.db_url = db_url
    
    def search(self, query, top_k=3):
        result = self.voyage_client.embed(
            texts=[query],
            model=EMBEDDING_MODEL,
            input_type="query"  # CRITICAL: marks this as a query for retrieval
        )
        query_embedding = result.embeddings[0]
        
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT book_name, chunk_index, content,
                   1 - (embedding <=> %s::vector) as similarity
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, top_k))
        
        results = []
        for row in cur.fetchall():
            results.append({
                'book_name': row[0],
                'chunk_index': row[1],
                'content': row[2],
                'similarity': float(row[3])
            })
        
        cur.close()
        conn.close()
        return results


def check_relevance(content, expected_keywords):
    """Check if content contains any of the expected keywords."""
    content_lower = content.lower()
    found = []
    missing = []
    
    for keyword in expected_keywords:
        if keyword.lower() in content_lower:
            found.append(keyword)
        else:
            missing.append(keyword)
    
    return len(found) > 0, found, missing


def run_tests():
    print("=" * 60)
    print("RETRIEVAL QUALITY TEST")
    print("=" * 60)
    print(f"Testing {len(TEST_CASES)} questions")
    print("Checking if top-3 chunks contain expected answer content")
    print()
    
    retriever = Retriever()
    
    passed = 0
    failed = 0
    failed_cases = []
    
    for i, (question, expected_keywords) in enumerate(TEST_CASES, 1):
        results = retriever.search(question, top_k=3)
        
        # Check if ANY of top 3 chunks contain relevant keywords
        any_relevant = False
        best_match = None
        
        for result in results:
            is_relevant, found, missing = check_relevance(result['content'], expected_keywords)
            if is_relevant:
                any_relevant = True
                if best_match is None or len(found) > len(best_match['found']):
                    best_match = {
                        'result': result,
                        'found': found,
                        'missing': missing
                    }
        
        if any_relevant:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"
            failed_cases.append({
                'question': question,
                'expected': expected_keywords,
                'top_result': results[0] if results else None,
                'all_results': results
            })
        
        # Progress
        print(f"[{i:3d}/100] {status}: {question[:50]}...")
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/100 ({passed}%)")
    print(f"Failed: {failed}/100 ({failed}%)")
    
    # Show failed cases
    if failed_cases:
        print()
        print("=" * 60)
        print("FAILED CASES (showing first 10)")
        print("=" * 60)
        
        for case in failed_cases[:10]:
            print(f"\nQuestion: {case['question']}")
            print(f"Expected keywords: {case['expected']}")
            if case['top_result']:
                print(f"Top result ({case['top_result']['book_name']}, sim={case['top_result']['similarity']:.4f}):")
                print(f"  {case['top_result']['content'][:300]}...")
            print("-" * 40)
    
    # Save full results to file
    with open("data/processed/retrieval_test_results.txt", "w", encoding="utf-8") as f:
        f.write(f"RETRIEVAL QUALITY TEST RESULTS\n")
        f.write(f"Passed: {passed}/100 ({passed}%)\n")
        f.write(f"Failed: {failed}/100 ({failed}%)\n\n")
        
        f.write("FAILED CASES:\n")
        f.write("=" * 60 + "\n")
        for case in failed_cases:
            f.write(f"\nQuestion: {case['question']}\n")
            f.write(f"Expected: {case['expected']}\n")
            for j, r in enumerate(case['all_results'], 1):
                f.write(f"\nResult {j} ({r['book_name']}, chunk {r['chunk_index']}, sim={r['similarity']:.4f}):\n")
                f.write(f"{r['content'][:500]}\n")
            f.write("-" * 60 + "\n")
    
    print(f"\nFull results saved to: data/processed/retrieval_test_results.txt")


if __name__ == "__main__":
    run_tests()
