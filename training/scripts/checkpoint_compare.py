#!/usr/bin/env python3
"""
Checkpoint Comparison for Model v3
Compares outputs from different training epochs to see progression.
"""

import torch
import json
from pathlib import Path

# Test questions for comparison
TEST_QUESTIONS = [
    "What is a qubit?",
    "What is superposition?",
    "What is quantum entanglement?",
    "How does a quantum gate work?",
    "What is measurement in quantum computing?",
]

def load_model(checkpoint_path, config_path, device):
    """Load model from checkpoint."""
    from model import Transformer
    
    with open(config_path) as f:
        config = json.load(f)
    
    model = Transformer(
        vocab_size=config["vocab_size"],
        dim=config["dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"],
        ff_dim=config["ff_dim"],
        dropout=config["dropout"],
        pad_token_id=config["pad_token_id"]
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

def generate_response(model, tokenizer, question, device, max_new_tokens=80):
    """Generate response for a question."""
    prompt = f"Q: {question}\nA:"
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], device=device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_tensor.size(1) >= 512:
                break
            
            logits = model(input_tensor)
            next_token_logits = logits[0, -1, :]
            
            # Temperature sampling
            probs = torch.softmax(next_token_logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
            
            # Stop on newline or special tokens
            if next_token.item() in [0, 1, 2]:
                break
    
    output_ids = input_tensor[0].tolist()
    response = tokenizer.decode(output_ids[len(input_ids):])
    
    # Clean up response
    response = response.split("\n")[0].strip()
    return response[:300]

def main():
    from tokenizers import Tokenizer
    
    device = torch.device("cpu")
    config_path = Path("model/config.json")
    tokenizer_path = Path("tokenizer.json")
    
    # Checkpoints to compare
    checkpoints = [
        ("Epoch 1", "model/checkpoint_epoch1.pt"),
        ("Epoch 5", "model/checkpoint_epoch5.pt"),
        ("Epoch 10", "model/checkpoint_epoch10.pt"),
    ]
    
    # Verify files exist
    print("=" * 70)
    print("CHECKPOINT COMPARISON: Training Progression")
    print("=" * 70)
    print()
    
    if not config_path.exists():
        print(f"ERROR: {config_path} not found")
        return
    
    if not tokenizer_path.exists():
        print(f"ERROR: {tokenizer_path} not found")
        return
    
    for name, path in checkpoints:
        if not Path(path).exists():
            print(f"ERROR: {path} not found")
            return
    
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    print("Loading checkpoints and generating responses...")
    print()
    
    # Generate responses for each question at each checkpoint
    for q_idx, question in enumerate(TEST_QUESTIONS, 1):
        print("=" * 70)
        print(f"Q{q_idx}: {question}")
        print("=" * 70)
        
        for name, path in checkpoints:
            model = load_model(path, config_path, device)
            response = generate_response(model, tokenizer, question, device)
            
            print(f"\n[{name}]")
            print(f"  {response}")
            
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print()
    
    # Summary guidance
    print("=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print()
    print("Look for these improvements across epochs:")
    print("  1. Less repetition (e.g., 'qubit qubit qubit')")
    print("  2. More complete sentences")
    print("  3. Better use of quantum terminology")
    print("  4. More coherent structure")
    print()
    print("Expected for 1.2M params:")
    print("  - Vocabulary improves across epochs")
    print("  - Coherence may plateau (model too small for reasoning)")
    print("  - No boilerplate phrases (data is clean)")

if __name__ == "__main__":
    main()
