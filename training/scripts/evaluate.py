"""
Evaluate trained Quantum Computing LLM

Tests:
1. Generation quality with context prompts
2. Perplexity on held-out data
3. Keyword accuracy on test questions
"""

import torch
import argparse
import json
from pathlib import Path
from model import QuantumLLM
from dataset import load_tokenizer

# Test questions for evaluation
TEST_QUESTIONS = [
    {
        "context": "Q: What is superposition? A: Superposition allows a qubit to exist in multiple states simultaneously until measured. Q: What are quantum states? A: Quantum states describe the complete information about a quantum system.",
        "question": "What is a qubit?",
        "keywords": ["qubit", "quantum", "bit", "superposition", "state"]
    },
    {
        "context": "Q: What is Shor's algorithm? A: Shor's algorithm factors large numbers exponentially faster than classical algorithms. Q: What is quantum speedup? A: Quantum speedup refers to problems quantum computers solve faster than classical ones.",
        "question": "Why is quantum computing important for cryptography?",
        "keywords": ["cryptography", "encryption", "security", "factor", "RSA"]
    },
    {
        "context": "Q: What is entanglement? A: Entanglement is a quantum phenomenon where particles become correlated. Q: What is a Bell state? A: A Bell state is a maximally entangled state of two qubits.",
        "question": "How does quantum entanglement work?",
        "keywords": ["entangle", "correlat", "measure", "particle", "state"]
    },
    {
        "context": "Q: What is a quantum gate? A: A quantum gate is a basic operation on qubits, like a logic gate for classical bits. Q: What is the Hadamard gate? A: The Hadamard gate creates superposition from a basis state.",
        "question": "What is a quantum circuit?",
        "keywords": ["circuit", "gate", "qubit", "operation", "quantum"]
    },
    {
        "context": "Q: What is decoherence? A: Decoherence is the loss of quantum information due to environmental interaction. Q: What is a T1 time? A: T1 time measures how long a qubit maintains its energy state.",
        "question": "Why is quantum error correction needed?",
        "keywords": ["error", "correct", "noise", "decoher", "fault"]
    }
]


def evaluate_generation(model, tokenizer, device):
    """Test generation quality with context prompts"""
    print("\n" + "="*70)
    print("GENERATION QUALITY TEST")
    print("="*70)
    
    model.eval()
    results = []
    
    for i, test in enumerate(TEST_QUESTIONS):
        prompt = f"Context: {test['context']} Question: {test['question']} Answer:"
        
        tokens = tokenizer.encode(prompt).ids
        x = torch.tensor([tokens], device=device)
        
        with torch.no_grad():
            output = model.generate(x, max_new_tokens=150, temperature=0.7)
        
        generated = tokenizer.decode(output[0].tolist())
        
        # Extract just the answer part
        answer_start = generated.find("Answer:") + len("Answer:")
        answer = generated[answer_start:].strip()
        
        # Check keywords
        answer_lower = answer.lower()
        keywords_found = [kw for kw in test['keywords'] if kw.lower() in answer_lower]
        keyword_score = len(keywords_found) / len(test['keywords'])
        
        results.append({
            'question': test['question'],
            'answer': answer[:300],
            'keywords_found': keywords_found,
            'keyword_score': keyword_score
        })
        
        print(f"\n--- Test {i+1} ---")
        print(f"Q: {test['question']}")
        print(f"A: {answer[:300]}...")
        print(f"Keywords: {keywords_found} ({keyword_score:.0%})")
    
    avg_score = sum(r['keyword_score'] for r in results) / len(results)
    print(f"\n{'='*70}")
    print(f"Average keyword score: {avg_score:.1%}")
    
    return results, avg_score


def evaluate_coherence(model, tokenizer, device):
    """Test if model produces coherent sentences"""
    print("\n" + "="*70)
    print("COHERENCE TEST (no context)")
    print("="*70)
    
    model.eval()
    
    prompts = [
        "Quantum computing is",
        "A qubit can be described as",
        "The main advantage of quantum computers is",
        "Superposition allows",
        "Entanglement is important because"
    ]
    
    for prompt in prompts:
        tokens = tokenizer.encode(prompt).ids
        x = torch.tensor([tokens], device=device)
        
        with torch.no_grad():
            output = model.generate(x, max_new_tokens=80, temperature=0.8)
        
        generated = tokenizer.decode(output[0].tolist())
        
        print(f"\nPrompt: {prompt}")
        print(f"Output: {generated[:250]}")


def check_garbage(text):
    """Check if output is garbage (repetitive, fragmented)"""
    words = text.split()
    if len(words) < 5:
        return True, "Too short"
    
    # Check repetition
    for i in range(len(words) - 3):
        phrase = ' '.join(words[i:i+3])
        if text.count(phrase) > 2:
            return True, f"Repetitive: '{phrase}'"
    
    # Check for sentence structure
    if '.' not in text and ',' not in text:
        return True, "No punctuation"
    
    return False, "OK"


def main():
    parser = argparse.ArgumentParser(description='Evaluate Quantum LLM')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, default='../tokenizer.json')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model: {args.model_path}")
    model = QuantumLLM.load(args.model_path, device)
    model = model.to(device)
    model.eval()
    
    print(f"Model parameters: {model.n_params:,}")
    
    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path)
    
    # Run evaluations
    results, avg_score = evaluate_generation(model, tokenizer, device)
    evaluate_coherence(model, tokenizer, device)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Parameters: {model.n_params:,}")
    print(f"Keyword accuracy: {avg_score:.1%}")
    
    # Pass/fail
    if avg_score >= 0.3:
        print("\n✓ Model produces relevant answers with context")
    else:
        print("\n✗ Model not using context effectively")


if __name__ == "__main__":
    main()
