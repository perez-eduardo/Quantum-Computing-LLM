"""
Quantum Computing LLM - Dataset and DataLoader
Handles Q&A pairs and book text for training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import csv
import json
from pathlib import Path
from typing import Optional
import random


class Tokenizer:
    """Wrapper for the trained BPE tokenizer"""
    def __init__(self, tokenizer_path: str):
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            self.tokenizer_data = json.load(f)
        
        self.vocab = self.tokenizer_data['model']['vocab']
        self.merges = self.tokenizer_data['model']['merges']
        
        # Build reverse vocab for decoding
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Special tokens
        self.pad_token_id = 0  # <pad>
        self.eos_token_id = 1  # <eos>
        self.unk_token_id = 2  # <unk>
        
        self.vocab_size = len(self.vocab)
        
        # Try to use tokenizers library if available
        try:
            from tokenizers import Tokenizer as HFTokenizer
            self.hf_tokenizer = HFTokenizer.from_file(tokenizer_path)
            self._use_hf = True
        except ImportError:
            self._use_hf = False
            print("Warning: tokenizers library not found. Using basic encoding.")

    def encode(self, text: str, add_eos: bool = True) -> list:
        """Encode text to token IDs"""
        if self._use_hf:
            ids = self.hf_tokenizer.encode(text).ids
        else:
            # Basic fallback - character level with vocab lookup
            ids = []
            for char in text:
                if char in self.vocab:
                    ids.append(self.vocab[char])
                else:
                    ids.append(self.unk_token_id)
        
        if add_eos:
            ids.append(self.eos_token_id)
        
        return ids

    def decode(self, ids: list) -> str:
        """Decode token IDs to text"""
        if self._use_hf:
            return self.hf_tokenizer.decode(ids)
        else:
            # Basic fallback
            tokens = [self.id_to_token.get(i, '<unk>') for i in ids]
            return ''.join(tokens)

    def __len__(self):
        return self.vocab_size


class QADataset(Dataset):
    """Dataset for Question-Answer pairs"""
    def __init__(
        self,
        csv_path: str,
        tokenizer: Tokenizer,
        max_length: int = 512,
        question_col: str = 'question',
        answer_col: str = 'answer',
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        print(f"Loading Q&A data from {csv_path}...")
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row.get(question_col, '').strip()
                a = row.get(answer_col, '').strip()
                if q and a:
                    self.data.append((q, a))
        
        print(f"Loaded {len(self.data):,} Q&A pairs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, answer = self.data[idx]
        
        # Format: "Q: {question}\nA: {answer}<eos>"
        text = f"Q: {question}\nA: {answer}"
        
        # Encode
        ids = self.tokenizer.encode(text, add_eos=True)
        
        # Truncate if needed
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        
        return torch.tensor(ids, dtype=torch.long)


class BookDataset(Dataset):
    """Dataset for book text (next token prediction)"""
    def __init__(
        self,
        txt_path: str,
        tokenizer: Tokenizer,
        max_length: int = 512,
        stride: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        print(f"Loading book data from {txt_path}...")
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Encode entire text
        print("Tokenizing book text...")
        self.all_ids = self.tokenizer.encode(text, add_eos=False)
        print(f"Total tokens: {len(self.all_ids):,}")
        
        # Calculate number of chunks
        self.n_chunks = max(1, (len(self.all_ids) - max_length) // stride + 1)
        print(f"Created {self.n_chunks:,} chunks (stride={stride})")

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.max_length
        
        # Get chunk
        ids = self.all_ids[start:end]
        
        # Pad if needed (last chunk might be shorter)
        if len(ids) < self.max_length:
            ids = ids + [self.tokenizer.pad_token_id] * (self.max_length - len(ids))
        
        return torch.tensor(ids, dtype=torch.long)


class CombinedDataset(Dataset):
    """Combines Q&A and Book datasets with configurable mixing"""
    def __init__(
        self,
        qa_dataset: QADataset,
        book_dataset: BookDataset,
        qa_weight: float = 0.8,
    ):
        self.qa_dataset = qa_dataset
        self.book_dataset = book_dataset
        self.qa_weight = qa_weight
        
        # Calculate total length based on the larger dataset
        self.qa_len = len(qa_dataset)
        self.book_len = len(book_dataset)
        self.total_len = self.qa_len + self.book_len
        
        print(f"Combined dataset: {self.qa_len:,} Q&A + {self.book_len:,} book chunks = {self.total_len:,} total")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < self.qa_len:
            return self.qa_dataset[idx]
        else:
            return self.book_dataset[idx - self.qa_len]


def collate_fn(batch, pad_token_id=0):
    """Collate function for variable length sequences"""
    max_len = max(len(x) for x in batch)
    
    padded = []
    for x in batch:
        if len(x) < max_len:
            padding = torch.full((max_len - len(x),), pad_token_id, dtype=torch.long)
            x = torch.cat([x, padding])
        padded.append(x)
    
    return torch.stack(padded)


def create_dataloaders(
    qa_path: str,
    book_path: str,
    tokenizer_path: str,
    batch_size: int = 32,
    max_length: int = 512,
    val_split: float = 0.05,
    num_workers: int = 4,
    seed: int = 42,
):
    """Create train and validation dataloaders"""
    
    # Load tokenizer
    tokenizer = Tokenizer(tokenizer_path)
    print(f"Loaded tokenizer with {len(tokenizer):,} tokens")
    
    # Create datasets
    qa_dataset = QADataset(qa_path, tokenizer, max_length)
    book_dataset = BookDataset(book_path, tokenizer, max_length)
    
    # Combine
    combined = CombinedDataset(qa_dataset, book_dataset)
    
    # Split
    total_len = len(combined)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len
    
    random.seed(seed)
    torch.manual_seed(seed)
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined, [train_len, val_len]
    )
    
    print(f"Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        pin_memory=True,
    )
    
    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    # Test dataset creation
    import sys
    
    # Default paths (adjust as needed)
    qa_path = "../data/combined_qa_final.csv"
    book_path = "../data/combined_books.txt"
    tokenizer_path = "../tokenizer.json"
    
    if len(sys.argv) > 1:
        qa_path = sys.argv[1]
    if len(sys.argv) > 2:
        book_path = sys.argv[2]
    if len(sys.argv) > 3:
        tokenizer_path = sys.argv[3]
    
    train_loader, val_loader, tokenizer = create_dataloaders(
        qa_path, book_path, tokenizer_path,
        batch_size=4,
        max_length=128,
        num_workers=0,
    )
    
    # Test batch
    for batch in train_loader:
        print(f"Batch shape: {batch.shape}")
        print(f"Sample decoded: {tokenizer.decode(batch[0].tolist()[:50])}...")
        break
