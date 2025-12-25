"""
Train BPE tokenizer on books + Q&A data
Run this FIRST before training the model
"""

import csv
import io
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

def load_texts(book_path, qa_paths):
    """Load all text data for tokenizer training"""
    texts = []
    
    # Load books
    print(f"Loading books from {book_path}...")
    with open(book_path, 'r', encoding='utf-8') as f:
        book_text = f.read()
    texts.append(book_text)
    print(f"  Book text: {len(book_text):,} characters")
    
    # Load Q&A CSVs
    for csv_path in qa_paths:
        print(f"Loading {csv_path}...")
        count = 0
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = [l for l in f if not l.startswith('#') and l.strip()]
        
        reader = csv.DictReader(io.StringIO(''.join(lines)))
        for row in reader:
            q = row.get('question', '').strip()
            a = row.get('answer', '').strip()
            c = row.get('context', '').strip()
            
            if q and a:
                # Format as training data will appear
                if c:
                    texts.append(f"Context: {c} Question: {q} Answer: {a}")
                else:
                    texts.append(f"Question: {q} Answer: {a}")
                count += 1
        
        print(f"  Loaded {count:,} Q&A pairs")
    
    print(f"\nTotal texts: {len(texts):,}")
    return texts


def train_tokenizer(texts, vocab_size=16384, output_path="tokenizer.json"):
    """Train BPE tokenizer"""
    print(f"\nTraining tokenizer with vocab_size={vocab_size}...")
    
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    
    # Pre-tokenizer: split on whitespace and punctuation
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Decoder
    tokenizer.decoder = decoders.ByteLevel()
    
    # Post-processor
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        show_progress=True,
        min_frequency=2
    )
    
    # Train
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    # Save
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")
    
    # Test
    print("\nTesting tokenizer...")
    test_texts = [
        "What is a qubit?",
        "Quantum superposition allows a qubit to exist in multiple states.",
        "Context: Q: What is entanglement? A: Entanglement correlates qubits. Question: How? Answer:",
    ]
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        print(f"  Original: {text[:60]}...")
        print(f"  Tokens: {len(encoded.ids)}")
        print(f"  Decoded: {decoded[:60]}...")
        print()
    
    # Show some quantum-specific tokens
    print("Sample vocabulary tokens:")
    vocab = tokenizer.get_vocab()
    quantum_tokens = [t for t in vocab.keys() if 'qubit' in t.lower() or 'quantum' in t.lower() or 'superpos' in t.lower()]
    for t in quantum_tokens[:20]:
        print(f"  {t}")
    
    return tokenizer


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train tokenizer')
    parser.add_argument('--book_path', type=str, default='../data/combined_books_cleaned.txt')
    parser.add_argument('--qa_paths', type=str, nargs='+', default=[
        '../data/claude_qa_context.csv',
        '../data/cot_qa_context.csv',
        '../data/stackexchange_qa_context.csv'
    ])
    parser.add_argument('--vocab_size', type=int, default=16384)
    parser.add_argument('--output', type=str, default='../tokenizer.json')
    
    args = parser.parse_args()
    
    # Load texts
    texts = load_texts(args.book_path, args.qa_paths)
    
    # Train tokenizer
    train_tokenizer(texts, args.vocab_size, args.output)
    
    print("\nDone! Now run training with:")
    print("  sbatch train_phase1.sh")


if __name__ == "__main__":
    main()
