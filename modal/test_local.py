import torch
import sys
sys.path.insert(0, '../training/scripts')
from model import QuantumLLM
from tokenizers import Tokenizer

# Load model
print("Loading model...")
model = QuantumLLM.load('../training/model/final_model.pt', 'cpu')
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = Tokenizer.from_file('../training/tokenizer/tokenizer.json')
print("Tokenizer loaded")

# Test prompt
context = "Q: What is superposition? A: Superposition allows a qubit to exist in multiple states simultaneously."
question = "What is a qubit?"
prompt = f"Context: {context} Question: {question} Answer:"

print(f"\nPrompt: {prompt[:80]}...")

# Generate
tokens = tokenizer.encode(prompt)
input_ids = torch.tensor([tokens.ids])

print("Generating...")
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=100, temperature=0.2, top_k=30)

result = tokenizer.decode(output[0].tolist())
print(f"\nFull output:\n{result}")