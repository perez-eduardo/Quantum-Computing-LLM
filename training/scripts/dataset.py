"""
Dataset classes for two-phase training:
Phase 1: Book pretraining (coherent prose generation)
Phase 2: Context Q&A fine-tuning (RAG context usage)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import csv
import random
from pathlib import Path


class BookDataset(Dataset):
    """
    Phase 1: Book pretraining dataset
    Chunks books into sequences for next-token prediction
    """
    def __init__(self, book_path, tokenizer, max_length=1024, stride=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunks = []
        
        print(f"Loading books from {book_path}...")
        with open(book_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"Book text: {len(text):,} characters")
        
        # Tokenize entire text
        tokens = tokenizer.encode(text).ids
        print(f"Total tokens: {len(tokens):,}")
        
        # Create overlapping chunks
        for i in range(0, len(tokens) - max_length, stride):
            chunk = tokens[i:i + max_length]
            if len(chunk) == max_length:
                self.chunks.append(chunk)
        
        print(f"Created {len(self.chunks):,} book chunks")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        tokens = self.chunks[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


class ContextQADataset(Dataset):
    """
    Phase 2: Context Q&A fine-tuning dataset
    Format: Context: [Q&A pairs] Question: [q] Answer: [a]
    """
    def __init__(self, csv_paths, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load from all CSV files
        for csv_path in csv_paths:
            print(f"Loading {csv_path}...")
            count = 0
            with open(csv_path, 'r', encoding='utf-8') as f:
                # Skip comment lines
                lines = [l for l in f if not l.startswith('#') and l.strip()]
            
            import io
            reader = csv.DictReader(io.StringIO(''.join(lines)))
            
            for row in reader:
                question = row.get('question', '').strip()
                answer = row.get('answer', '').strip()
                context = row.get('context', '').strip()
                
                if question and answer:
                    self.examples.append({
                        'question': question,
                        'answer': answer,
                        'context': context
                    })
                    count += 1
            
            print(f"  Loaded {count:,} examples")
        
        print(f"Total examples: {len(self.examples):,}")
        random.shuffle(self.examples)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Format: Context: [context] Question: [q] Answer: [a]
        if ex['context']:
            text = f"Context: {ex['context']} Question: {ex['question']} Answer: {ex['answer']}"
        else:
            text = f"Question: {ex['question']} Answer: {ex['answer']}"
        
        # Tokenize
        tokens = self.tokenizer.encode(text).ids
        
        # Truncate if needed
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Pad if needed
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        x = tokens[:-1]
        y = tokens[1:]
        
        return x, y


class CombinedDataset(Dataset):
    """
    Combined dataset for mixed training (books + Q&A)
    Uses weighted sampling to control mix ratio
    """
    def __init__(self, book_dataset, qa_dataset, book_weight=0.3):
        self.book_dataset = book_dataset
        self.qa_dataset = qa_dataset
        self.book_weight = book_weight
        
        # Calculate effective length
        self.book_len = len(book_dataset)
        self.qa_len = len(qa_dataset)
        self.total_len = self.book_len + self.qa_len
        
        print(f"Combined dataset: {self.book_len:,} book + {self.qa_len:,} Q&A = {self.total_len:,} total")
        print(f"Book weight: {book_weight:.0%}")
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        # Probabilistic selection based on weight
        if random.random() < self.book_weight:
            return self.book_dataset[idx % self.book_len]
        else:
            return self.qa_dataset[idx % self.qa_len]


def load_tokenizer(path):
    """Load tokenizers.Tokenizer from JSON file"""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def create_dataloaders(
    tokenizer_path,
    book_path=None,
    qa_csv_paths=None,
    max_length=1024,
    batch_size=8,
    phase=1,
    book_weight=0.3,
    num_workers=0
):
    """
    Create dataloaders for training
    
    phase=1: Book pretraining only
    phase=2: Context Q&A fine-tuning only
    phase=3: Combined (books + Q&A)
    """
    tokenizer = load_tokenizer(tokenizer_path)
    
    if phase == 1:
        # Phase 1: Books only
        assert book_path, "book_path required for phase 1"
        dataset = BookDataset(book_path, tokenizer, max_length)
        
    elif phase == 2:
        # Phase 2: Context Q&A only
        assert qa_csv_paths, "qa_csv_paths required for phase 2"
        dataset = ContextQADataset(qa_csv_paths, tokenizer, max_length)
        
    elif phase == 3:
        # Phase 3: Combined
        assert book_path and qa_csv_paths, "Both book_path and qa_csv_paths required for phase 3"
        book_ds = BookDataset(book_path, tokenizer, max_length)
        qa_ds = ContextQADataset(qa_csv_paths, tokenizer, max_length)
        dataset = CombinedDataset(book_ds, qa_ds, book_weight)
    
    else:
        raise ValueError(f"Unknown phase: {phase}")
    
    # Split train/val (95/5)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    # Test dataset creation
    import sys
    
    tokenizer_path = "../tokenizer.json"
    book_path = "../data/combined_books_cleaned.txt"
    qa_paths = [
        "../data/claude_qa_context.csv",
        "../data/cot_qa_context.csv",
        "../data/stackexchange_qa_context.csv"
    ]
    
    print("Testing Phase 1 (Books)...")
    train_loader, val_loader, tok = create_dataloaders(
        tokenizer_path, book_path=book_path, phase=1, batch_size=4
    )
    x, y = next(iter(train_loader))
    print(f"Batch shape: {x.shape}")
    
    print("\nTesting Phase 2 (Context Q&A)...")
    train_loader, val_loader, tok = create_dataloaders(
        tokenizer_path, qa_csv_paths=qa_paths, phase=2, batch_size=4
    )
    x, y = next(iter(train_loader))
    print(f"Batch shape: {x.shape}")
    
    # Decode a sample
    sample = tok.decode(x[0].tolist())
    print(f"Sample: {sample[:500]}...")
