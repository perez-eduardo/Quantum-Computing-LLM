"""
Quantum Computing BPE Tokenizer Training Script

Trains a custom BPE tokenizer on your quantum computing corpus.
Run locally (not on HPC).

Usage:
    python train_tokenizer.py --qa combined_qa.csv --books combined_books.txt --output tokenizer/

Requirements:
    pip install tokenizers pandas
"""

import argparse
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import pandas as pd


def load_qa_texts(qa_path: str) -> list[str]:
    """Load Q&A pairs from CSV and combine into training texts."""
    print(f"Loading Q&A data from {qa_path}...")
    df = pd.read_csv(qa_path)
    
    texts = []
    for _, row in df.iterrows():
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        if q and a:
            texts.append(f"{q}\n{a}")
    
    print(f"  Loaded {len(texts):,} Q&A pairs")
    return texts


def load_book_text(book_path: str) -> list[str]:
    """Load book text and split into chunks for training."""
    print(f"Loading book text from {book_path}...")
    with open(book_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split into paragraphs (double newline separated)
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    print(f"  Loaded {len(chunks):,} text chunks")
    return chunks


def train_tokenizer(
    texts: list[str],
    vocab_size: int = 16000,
    min_frequency: int = 2
) -> Tokenizer:
    """Train a BPE tokenizer on the provided texts."""
    print(f"\nTraining BPE tokenizer (vocab_size={vocab_size})...")
    
    # Initialize BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # Pre-tokenizer: split on whitespace and punctuation
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Decoder for proper detokenization
    tokenizer.decoder = decoders.ByteLevel()
    
    # Post-processor for byte-level
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Define special tokens
    special_tokens = ["<pad>", "<eos>", "<unk>"]
    
    # Trainer configuration
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )
    
    # Train from iterator
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    print(f"  Vocabulary size: {tokenizer.get_vocab_size():,}")
    
    return tokenizer


def test_tokenizer(tokenizer: Tokenizer):
    """Run some test encodings to verify tokenizer works."""
    print("\n" + "=" * 50)
    print("TOKENIZER TEST")
    print("=" * 50)
    
    test_sentences = [
        "What is a qubit?",
        "Quantum entanglement allows two qubits to be correlated.",
        "The Hadamard gate creates superposition.",
        "Apply a CNOT gate to entangle the qubits.",
        "Superposition means a qubit can be in both states simultaneously.",
    ]
    
    for sentence in test_sentences:
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded.ids)
        
        print(f"\nOriginal:  {sentence}")
        print(f"Tokens:    {encoded.tokens}")
        print(f"IDs:       {encoded.ids}")
        print(f"Decoded:   {decoded}")


def print_special_token_ids(tokenizer: Tokenizer):
    """Print IDs of special and domain tokens."""
    print("\n" + "=" * 50)
    print("SPECIAL TOKEN IDS")
    print("=" * 50)
    
    tokens_to_check = [
        "<pad>", "<eos>", "<unk>",
    ]
    
    vocab = tokenizer.get_vocab()
    for token in tokens_to_check:
        token_id = vocab.get(token, "NOT FOUND")
        print(f"  {token}: {token_id}")


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer for Quantum Computing LLM")
    parser.add_argument("--qa", required=True, help="Path to combined_qa.csv")
    parser.add_argument("--books", required=True, help="Path to combined_books.txt")
    parser.add_argument("--output", default="tokenizer", help="Output directory for tokenizer")
    parser.add_argument("--vocab-size", type=int, default=16000, help="Vocabulary size")
    parser.add_argument("--min-freq", type=int, default=2, help="Minimum token frequency")
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not Path(args.qa).exists():
        print(f"Error: Q&A file not found: {args.qa}")
        return 1
    if not Path(args.books).exists():
        print(f"Error: Books file not found: {args.books}")
        return 1
    
    # Load data
    qa_texts = load_qa_texts(args.qa)
    book_texts = load_book_text(args.books)
    all_texts = qa_texts + book_texts
    
    print(f"\nTotal training texts: {len(all_texts):,}")
    
    # Train tokenizer
    tokenizer = train_tokenizer(
        texts=all_texts,
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq
    )
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tokenizer
    output_path = output_dir / "tokenizer.json"
    tokenizer.save(str(output_path))
    print(f"\nTokenizer saved to: {output_path}")
    
    # Print stats and run tests
    print_special_token_ids(tokenizer)
    test_tokenizer(tokenizer)
    
    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Tokenizer ready at: {output_path}")
    print(f"Copy this file to HPC when setting up training.")
    
    return 0


if __name__ == "__main__":
    exit(main())