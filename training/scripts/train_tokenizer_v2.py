"""
Train BPE Tokenizer v2

Trains a custom BPE tokenizer on Claude Q&A + Stack Exchange + Books.
Replaces the old tokenizer trained on ChatGPT garbage.

Run on HPC:
    cd ~/hpc-share/quantum-llm
    source venv/bin/activate
    python scripts/train_tokenizer_v2.py
"""

import pandas as pd
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


def load_qa_texts(qa_path: Path) -> list:
    """Load Q&A pairs and format for training."""
    print(f"  Loading {qa_path.name}...")
    df = pd.read_csv(qa_path)
    
    texts = []
    for _, row in df.iterrows():
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        if q and a:
            # Format as training will see it
            texts.append(f"Q: {q}\nA: {a}")
    
    print(f"    Loaded {len(texts):,} Q&A pairs")
    return texts


def load_book_text(book_path: Path) -> list:
    """Load book text and split into chunks."""
    print(f"  Loading {book_path.name}...")
    with open(book_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split into paragraphs
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    print(f"    Loaded {len(chunks):,} text chunks")
    print(f"    Total words: {len(text.split()):,}")
    return chunks


def train_tokenizer(texts: list, vocab_size: int = 16000, min_frequency: int = 2) -> Tokenizer:
    """Train BPE tokenizer."""
    print(f"\n  Training BPE tokenizer (vocab_size={vocab_size})...")
    
    # Initialize BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # Pre-tokenizer: split on whitespace and punctuation
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Decoder for proper detokenization
    tokenizer.decoder = decoders.ByteLevel()
    
    # Post-processor
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Special tokens
    special_tokens = ["<pad>", "<eos>", "<unk>"]
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )
    
    # Train
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    print(f"    Vocabulary size: {tokenizer.get_vocab_size():,}")
    
    return tokenizer


def test_tokenizer(tokenizer: Tokenizer):
    """Test tokenizer with sample sentences."""
    print("\n" + "-" * 40)
    print("TOKENIZER TEST")
    print("-" * 40)
    
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
        
        print(f"\n  Original: {sentence}")
        print(f"  Tokens:   {len(encoded.ids)} -> {encoded.tokens[:10]}...")
        print(f"  Decoded:  {decoded}")


def check_special_tokens(tokenizer: Tokenizer):
    """Verify special token IDs."""
    print("\n" + "-" * 40)
    print("SPECIAL TOKEN IDS")
    print("-" * 40)
    
    vocab = tokenizer.get_vocab()
    expected = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    
    all_correct = True
    for token, expected_id in expected.items():
        actual_id = vocab.get(token, "NOT FOUND")
        status = "OK" if actual_id == expected_id else "WRONG"
        if actual_id != expected_id:
            all_correct = False
        print(f"  {token}: {actual_id} ({status})")
    
    return all_correct


def check_quantum_terms(tokenizer: Tokenizer):
    """Check if key quantum terms are efficient tokens."""
    print("\n" + "-" * 40)
    print("QUANTUM TERM EFFICIENCY")
    print("-" * 40)
    
    terms = [
        "qubit", "quantum", "superposition", "entanglement",
        "Hadamard", "CNOT", "gate", "measurement", "decoherence",
        "algorithm", "circuit", "state"
    ]
    
    for term in terms:
        encoded = tokenizer.encode(term)
        n_tokens = len(encoded.ids)
        efficiency = "single token" if n_tokens == 1 else f"{n_tokens} tokens"
        print(f"  {term}: {efficiency}")


def main():
    data_dir = Path.home() / "hpc-share/quantum-llm/data"
    output_dir = Path.home() / "hpc-share/quantum-llm"
    
    qa_path = data_dir / "combined_qa.csv"
    book_path = data_dir / "combined_books.txt"
    output_path = output_dir / "tokenizer.json"
    
    print("=" * 60)
    print("TRAIN BPE TOKENIZER v2")
    print("=" * 60)
    print("\nThis replaces the old tokenizer trained on ChatGPT garbage.")
    
    # Validate inputs
    print("\n[1] Checking input files...")
    if not qa_path.exists():
        print(f"  ERROR: {qa_path} not found")
        print("  Run combine_qa_v2.py first!")
        return 1
    if not book_path.exists():
        print(f"  ERROR: {book_path} not found")
        return 1
    print("  All files found.")
    
    # Load data
    print("\n[2] Loading training data...")
    qa_texts = load_qa_texts(qa_path)
    book_texts = load_book_text(book_path)
    
    all_texts = qa_texts + book_texts
    print(f"\n  Total training texts: {len(all_texts):,}")
    
    # Train tokenizer
    print("\n[3] Training tokenizer...")
    tokenizer = train_tokenizer(all_texts, vocab_size=16000, min_frequency=2)
    
    # Save
    print(f"\n[4] Saving to {output_path}...")
    
    # Backup old tokenizer if exists
    if output_path.exists():
        backup_path = output_dir / "tokenizer_old.json"
        print(f"  Backing up old tokenizer to {backup_path.name}")
        output_path.rename(backup_path)
    
    tokenizer.save(str(output_path))
    print(f"  Saved: {output_path}")
    
    # Verify
    print("\n[5] Verification...")
    check_special_tokens(tokenizer)
    check_quantum_terms(tokenizer)
    test_tokenizer(tokenizer)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Q&A texts:       {len(qa_texts):,}")
    print(f"  Book chunks:     {len(book_texts):,}")
    print(f"  Vocab size:      {tokenizer.get_vocab_size():,}")
    print(f"  Output:          {output_path}")
    
    print("\n" + "=" * 60)
    print("NEXT STEP")
    print("=" * 60)
    print("  python scripts/filter_long_examples.py")
    
    return 0


if __name__ == "__main__":
    exit(main())
