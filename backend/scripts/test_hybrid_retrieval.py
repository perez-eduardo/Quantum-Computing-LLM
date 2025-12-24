"""
Test hybrid retrieval vs semantic-only.
Compares pass rates for both methods.

Run from project root: python backend/scripts/test_hybrid_retrieval.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from hybrid_retrieval import HybridRetriever


# Same test cases as before
TEST_CASES = [
    {"question": "What is a qubit?", "keywords": ["two-level", "quantum bit", "superposition", "|0⟩", "|1⟩"]},
    {"question": "What is superposition?", "keywords": ["both states", "simultaneously", "linear combination", "probability"]},
    {"question": "What is quantum entanglement?", "keywords": ["correlated", "non-local", "Bell", "EPR"]},
    {"question": "What is quantum interference?", "keywords": ["amplitude", "constructive", "destructive", "probability"]},
    {"question": "What is a quantum gate?", "keywords": ["unitary", "operation", "transform", "qubit"]},
    {"question": "What is the Bloch sphere?", "keywords": ["representation", "single qubit", "sphere", "visualization"]},
    {"question": "What is quantum measurement?", "keywords": ["collapse", "eigenvalue", "probability", "observable"]},
    {"question": "What is the no-cloning theorem?", "keywords": ["cannot copy", "unknown", "quantum state"]},
    {"question": "What is quantum decoherence?", "keywords": ["environment", "loss", "classical", "noise"]},
    {"question": "What is a quantum circuit?", "keywords": ["gates", "sequence", "qubits", "operations"]},
    {"question": "What is quantum parallelism?", "keywords": ["multiple", "simultaneously", "superposition", "exponential"]},
    {"question": "What is a tensor product?", "keywords": ["⊗", "combine", "composite", "Hilbert space"]},
    {"question": "What is the computational basis?", "keywords": ["|0⟩", "|1⟩", "basis states", "standard basis", "orthonormal"]},
    {"question": "What is quantum phase?", "keywords": ["angle", "complex", "relative", "global"]},
    {"question": "What is a pure state?", "keywords": ["single", "ket", "vector", "superposition"]},
    {"question": "What is a mixed state?", "keywords": ["ensemble", "density matrix", "statistical", "classical mixture"]},
    {"question": "What is the Born rule?", "keywords": ["probability", "amplitude squared", "|ψ|²", "measurement"]},
    {"question": "What is a Hilbert space?", "keywords": ["vector space", "inner product", "complete", "quantum states"]},
    {"question": "What is quantum tunneling?", "keywords": ["barrier", "probability", "penetrate", "classically forbidden"]},
    {"question": "What is a quantum oracle?", "keywords": ["black box", "function", "query", "algorithm"]},
    {"question": "What is the Hadamard gate?", "keywords": ["H gate", "superposition", "1/√2", "equal"]},
    {"question": "What is the Pauli-X gate?", "keywords": ["NOT", "bit flip", "σx", "X gate"]},
    {"question": "What is the Pauli-Y gate?", "keywords": ["Y gate", "σy", "rotation", "imaginary"]},
    {"question": "What is the Pauli-Z gate?", "keywords": ["phase flip", "σz", "Z gate", "|1⟩"]},
    {"question": "What is the CNOT gate?", "keywords": ["controlled", "NOT", "two-qubit", "entanglement"]},
    {"question": "What is the SWAP gate?", "keywords": ["exchange", "qubits", "swap"]},
    {"question": "What is the Toffoli gate?", "keywords": ["CCNOT", "three-qubit", "controlled-controlled", "universal"]},
    {"question": "What is the T gate?", "keywords": ["π/8", "π/4", "phase", "non-Clifford"]},
    {"question": "What is the S gate?", "keywords": ["π/4", "π/2", "phase", "Clifford"]},
    {"question": "What is a controlled gate?", "keywords": ["control", "target", "conditional", "two-qubit"]},
    {"question": "What is the Fredkin gate?", "keywords": ["controlled swap", "CSWAP", "three-qubit", "reversible"]},
    {"question": "What are universal gates?", "keywords": ["any operation", "complete set", "approximate", "CNOT", "single-qubit"]},
    {"question": "What is a rotation gate?", "keywords": ["Rx", "Ry", "Rz", "angle", "axis"]},
    {"question": "What is the phase gate?", "keywords": ["diagonal", "phase shift", "P gate"]},
    {"question": "How do you create entanglement with gates?", "keywords": ["CNOT", "Hadamard", "Bell state"]},
    {"question": "What is gate decomposition?", "keywords": ["break down", "universal", "elementary", "compile"]},
    {"question": "What is gate fidelity?", "keywords": ["accuracy", "error", "ideal", "quality"]},
    {"question": "What is a parametrized gate?", "keywords": ["variational", "angle", "parameter", "VQE"]},
    {"question": "What is the CZ gate?", "keywords": ["controlled-Z", "phase", "two-qubit"]},
    {"question": "What is the iSWAP gate?", "keywords": ["swap", "phase", "i", "superconducting"]},
    {"question": "What is Shor's algorithm?", "keywords": ["factoring", "period finding", "exponential speedup", "RSA"]},
    {"question": "What is Grover's algorithm?", "keywords": ["search", "quadratic speedup", "amplitude amplification", "√N"]},
    {"question": "What is the Deutsch algorithm?", "keywords": ["oracle", "constant", "balanced", "first quantum"]},
    {"question": "What is the Deutsch-Jozsa algorithm?", "keywords": ["constant", "balanced", "exponential speedup", "oracle"]},
    {"question": "What is quantum phase estimation?", "keywords": ["eigenvalue", "unitary", "phase", "precision"]},
    {"question": "What is the quantum Fourier transform?", "keywords": ["QFT", "Fourier", "superposition", "phase"]},
    {"question": "What is amplitude amplification?", "keywords": ["Grover", "boost", "probability", "marked"]},
    {"question": "What is the Bernstein-Vazirani algorithm?", "keywords": ["hidden string", "dot product", "oracle", "linear"]},
    {"question": "What is Simon's algorithm?", "keywords": ["period", "hidden subgroup", "exponential", "oracle"]},
    {"question": "What is the HHL algorithm?", "keywords": ["linear systems", "matrix inversion", "exponential speedup"]},
    {"question": "What is VQE?", "keywords": ["variational", "eigensolver", "ground state", "chemistry"]},
    {"question": "What is QAOA?", "keywords": ["optimization", "combinatorial", "approximate", "MaxCut", "variational"]},
    {"question": "What is quantum walk?", "keywords": ["walk", "graph", "superposition", "speedup"]},
    {"question": "What is adiabatic quantum computing?", "keywords": ["ground state", "Hamiltonian", "slow", "evolution"]},
    {"question": "What is quantum annealing?", "keywords": ["optimization", "D-Wave", "Ising", "thermal"]},
    {"question": "What is quantum machine learning?", "keywords": ["QML", "kernel", "classification", "speedup"]},
    {"question": "What is the swap test?", "keywords": ["overlap", "inner product", "fidelity", "comparison"]},
    {"question": "What is quantum counting?", "keywords": ["Grover", "number of solutions", "phase estimation"]},
    {"question": "What is the quantum approximate counting?", "keywords": ["estimation", "solutions", "approximate"]},
    {"question": "What is quantum simulation?", "keywords": ["Hamiltonian", "physical system", "Feynman", "simulate"]},
    {"question": "What is quantum error correction?", "keywords": ["redundancy", "syndrome", "logical qubit", "protect"]},
    {"question": "What is the bit flip code?", "keywords": ["three qubit", "X error", "majority", "repetition"]},
    {"question": "What is the phase flip code?", "keywords": ["Z error", "three qubit", "Hadamard"]},
    {"question": "What is the Shor code?", "keywords": ["nine qubit", "bit flip", "phase flip", "first"]},
    {"question": "What is the Steane code?", "keywords": ["seven qubit", "CSS", "fault-tolerant"]},
    {"question": "What is a stabilizer code?", "keywords": ["Pauli", "stabilizer", "generator", "syndrome"]},
    {"question": "What is the surface code?", "keywords": ["2D", "lattice", "threshold", "topological"]},
    {"question": "What is fault-tolerant quantum computing?", "keywords": ["error", "threshold", "arbitrary", "concatenation"]},
    {"question": "What is the threshold theorem?", "keywords": ["error rate", "below threshold", "arbitrary accuracy"]},
    {"question": "What are Pauli errors?", "keywords": ["X", "Y", "Z", "bit flip", "phase flip"]},
    {"question": "What is a logical qubit?", "keywords": ["encoded", "physical qubits", "protected", "error correction"]},
    {"question": "What is syndrome measurement?", "keywords": ["error detection", "ancilla", "stabilizer", "parity"]},
    {"question": "What is a CSS code?", "keywords": ["Calderbank-Shor-Steane", "classical codes", "X and Z"]},
    {"question": "What is quantum fault tolerance?", "keywords": ["error propagation", "transversal", "threshold"]},
    {"question": "What is magic state distillation?", "keywords": ["T gate", "non-Clifford", "purification", "fault-tolerant"]},
    {"question": "What is a superconducting qubit?", "keywords": ["Josephson", "transmon", "microwave", "cryogenic"]},
    {"question": "What is an ion trap qubit?", "keywords": ["trapped ion", "laser", "electromagnetic", "long coherence"]},
    {"question": "What is a photonic qubit?", "keywords": ["photon", "polarization", "optical", "linear optics"]},
    {"question": "What is a topological qubit?", "keywords": ["Majorana", "anyons", "braiding", "protected"]},
    {"question": "What is quantum coherence time?", "keywords": ["T1", "T2", "decoherence", "how long"]},
    {"question": "What is T1 relaxation?", "keywords": ["energy decay", "amplitude damping", "excited to ground"]},
    {"question": "What is T2 dephasing?", "keywords": ["phase coherence", "dephasing", "T2*"]},
    {"question": "What is a dilution refrigerator?", "keywords": ["millikelvin", "cooling", "helium", "cryogenic"]},
    {"question": "What is quantum volume?", "keywords": ["benchmark", "metric", "IBM", "depth and width"]},
    {"question": "What is NISQ?", "keywords": ["noisy", "intermediate-scale", "near-term", "50-100 qubits"]},
    {"question": "What is quantum supremacy?", "keywords": ["advantage", "classical impossible", "Google", "Sycamore"]},
    {"question": "What are neutral atom qubits?", "keywords": ["atom", "optical tweezer", "Rydberg", "array"]},
    {"question": "What is a transmon qubit?", "keywords": ["superconducting", "charge", "anharmonic", "Josephson"]},
    {"question": "What is readout error?", "keywords": ["measurement", "assignment", "fidelity", "distinguish"]},
    {"question": "What is gate error rate?", "keywords": ["fidelity", "infidelity", "1-F", "benchmark"]},
    {"question": "What is a density matrix?", "keywords": ["ρ", "mixed state", "trace", "positive"]},
    {"question": "What is the trace operation?", "keywords": ["sum diagonal", "Tr", "expectation"]},
    {"question": "What is the partial trace?", "keywords": ["subsystem", "reduced", "trace out", "entanglement"]},
    {"question": "What is operator notation?", "keywords": ["bra-ket", "Dirac", "|ψ⟩", "⟨φ|"]},
    {"question": "What is an eigenstate?", "keywords": ["eigenvector", "eigenvalue", "measurement outcome", "definite"]},
    {"question": "What is a unitary matrix?", "keywords": ["U†U=I", "invertible", "norm preserving", "reversible"]},
    {"question": "What is a Hermitian operator?", "keywords": ["self-adjoint", "H†=H", "observable", "real eigenvalues"]},
    {"question": "What is the commutator?", "keywords": ["[A,B]", "AB-BA", "compatible", "simultaneous"]},
    {"question": "What is fidelity?", "keywords": ["similarity", "overlap", "closeness", "F(ρ,σ)"]},
    {"question": "What is the Pauli group?", "keywords": ["I,X,Y,Z", "group", "multiplication", "stabilizer"]},
]


def check_relevance(content: str, keywords: list) -> bool:
    """Check if content contains any expected keyword."""
    content_lower = content.lower()
    for kw in keywords:
        if kw.lower() in content_lower:
            return True
    return False


def run_test(retriever: HybridRetriever, method: str, top_k: int = 3):
    """Run test suite with specified method."""
    passed = 0
    failed = 0
    failures = []

    for i, tc in enumerate(TEST_CASES, 1):
        question = tc["question"]
        keywords = tc["keywords"]

        if method == "hybrid":
            results = retriever.search(question, top_k=top_k)
        elif method == "semantic":
            results = retriever.search_semantic_only(question, top_k=top_k)
        elif method == "keyword":
            results = retriever.search_keyword_only(question, top_k=top_k)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Check if any result contains expected content
        found = False
        for r in results:
            if check_relevance(r.get('content', ''), keywords):
                found = True
                break

        if found:
            passed += 1
        else:
            failed += 1
            failures.append({
                "question": question,
                "keywords": keywords,
                "top_result": results[0] if results else None
            })

    return passed, failed, failures


def main():
    print("=" * 60)
    print("HYBRID RETRIEVAL COMPARISON TEST")
    print("=" * 60)
    print(f"Testing {len(TEST_CASES)} questions\n")

    retriever = HybridRetriever()

    # Test semantic only
    print("[1] Testing SEMANTIC ONLY...")
    sem_pass, sem_fail, sem_failures = run_test(retriever, "semantic")
    print(f"    Result: {sem_pass}/{len(TEST_CASES)} ({100*sem_pass/len(TEST_CASES):.0f}%)")

    # Test keyword only
    print("\n[2] Testing KEYWORD ONLY...")
    kw_pass, kw_fail, kw_failures = run_test(retriever, "keyword")
    print(f"    Result: {kw_pass}/{len(TEST_CASES)} ({100*kw_pass/len(TEST_CASES):.0f}%)")

    # Test hybrid
    print("\n[3] Testing HYBRID (α=0.5)...")
    hyb_pass, hyb_fail, hyb_failures = run_test(retriever, "hybrid")
    print(f"    Result: {hyb_pass}/{len(TEST_CASES)} ({100*hyb_pass/len(TEST_CASES):.0f}%)")

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Semantic only: {sem_pass}/100 ({sem_pass}%)")
    print(f"  Keyword only:  {kw_pass}/100 ({kw_pass}%)")
    print(f"  Hybrid:        {hyb_pass}/100 ({hyb_pass}%)")

    # Show hybrid failures
    if hyb_failures:
        print(f"\n{'=' * 60}")
        print(f"HYBRID FAILURES ({len(hyb_failures)})")
        print("=" * 60)
        for f in hyb_failures[:10]:
            print(f"\nQuestion: {f['question']}")
            print(f"Expected: {f['keywords']}")
            if f['top_result']:
                src = f['top_result'].get('book_name', 'unknown')
                content = f['top_result'].get('content', '')[:200]
                print(f"Got ({src}): {content}...")
            print("-" * 40)


if __name__ == "__main__":
    main()
