"""
Quick sanity check: 10 sample questions, eyeball the outputs.
Run from: ~/hpc-share/quantum-llm/scripts/
"""

import torch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import QuantumLLM

# Paths
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "model" / "final_model.pt"
CONFIG_PATH = BASE_DIR / "model" / "config.json"
TOKENIZER_PATH = BASE_DIR / "tokenizer.json"


def load_tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


def main():
    print("=" * 60)
    print("SANITY CHECK: Quick Model Quality Inspection")
    print("=" * 60)
    
    # Load config
    print("\nLoading config...")
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    print(f"  Config: {config}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    
    # Load model
    print("\nLoading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    model = QuantumLLM(
        vocab_size=config["vocab_size"],
        dim=config["dim"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        ff_dim=config["ff_dim"],
        max_seq_len=config["max_seq_len"],
        dropout=0.0
    ).to(device)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"  Parameters: {model.num_parameters():,}")
    
    # Test questions
    questions = [
        "What is a qubit?",
        "What is superposition?",
        "What is entanglement?",
        "What is a Hadamard gate?",
        "What is a CNOT gate?",
        "How is a qubit different from a classical bit?",
        "What is quantum computing used for?",
        "What is decoherence?",
        "What is a quantum gate?",
        "What is measurement in quantum computing?",
    ]
    
    print("\n" + "=" * 60)
    print("GENERATING RESPONSES")
    print("=" * 60)
    
    for i, q in enumerate(questions, 1):
        prompt = f"Question: {q}\nAnswer:"
        print(f"\n[{i}/10] {q}")
        print("-" * 40)
        
        # Encode prompt
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], device=device)
        
        # Generate using model's built-in method
        output_ids = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            eos_token_id=1
        )
        
        # Decode
        response = tokenizer.decode(output_ids[0].tolist())
        
        # Extract answer part
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
        else:
            answer = response
        
        print(answer[:500])
        print()
    
    print("=" * 60)
    print("SANITY CHECK COMPLETE")
    print("=" * 60)
    print("\nLook for:")
    print("  - Does output contain quantum terminology?")
    print("  - Is it complete gibberish or somewhat coherent?")
    print("  - Does it stay on topic?")
    print("  - Any obvious failure modes (repetition, empty, etc)?")


if __name__ == "__main__":
    main()
