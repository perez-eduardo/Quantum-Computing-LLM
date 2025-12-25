"""
Two-Phase Training Script for Quantum Computing LLM

Phase 1: Pretrain on books (learn coherent prose generation)
Phase 2: Fine-tune on context Q&A (learn to use RAG context)

Usage:
    # Phase 1: Book pretraining
    python train.py --phase 1 --epochs 5
    
    # Phase 2: Context Q&A fine-tuning (loads phase 1 checkpoint)
    python train.py --phase 2 --epochs 10 --checkpoint model/phase1_final.pt
"""

import os
import sys
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path

from model import QuantumLLM, DEFAULT_CONFIG, save_config
from dataset import create_dataloaders


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine learning rate schedule with warmup"""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train_epoch(model, train_loader, optimizer, scaler, device, epoch, max_lr, min_lr, warmup_steps, max_steps, global_step):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        # Update learning rate
        lr = get_lr(global_step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass with mixed precision
        with autocast():
            logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        global_step += 1
        
        # Log progress
        if batch_idx % 50 == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * x.shape[0] / elapsed
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | LR: {lr:.2e} | "
                  f"Speed: {samples_per_sec:.1f} samples/s")
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss, global_step


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        with autocast():
            logits, loss = model(x, y)
        total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser(description='Train Quantum Computing LLM')
    
    # Phase selection
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3],
                        help='Training phase: 1=books, 2=context Q&A, 3=combined')
    
    # Data paths
    parser.add_argument('--tokenizer_path', type=str, default='../tokenizer.json')
    parser.add_argument('--book_path', type=str, default='../data/combined_books_cleaned.txt')
    parser.add_argument('--qa_paths', type=str, nargs='+', default=[
        '../data/claude_qa_context.csv',
        '../data/cot_qa_context.csv',
        '../data/stackexchange_qa_context.csv'
    ])
    
    # Model
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='../model')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_lr', type=float, default=3e-4)
    parser.add_argument('--min_lr', type=float, default=3e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--book_weight', type=float, default=0.3,
                        help='Weight for books in combined training (phase 3)')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    print(f"\n{'='*60}")
    print(f"PHASE {args.phase} TRAINING")
    print(f"{'='*60}")
    
    train_loader, val_loader, tokenizer = create_dataloaders(
        tokenizer_path=args.tokenizer_path,
        book_path=args.book_path if args.phase in [1, 3] else None,
        qa_csv_paths=args.qa_paths if args.phase in [2, 3] else None,
        max_length=args.max_length,
        batch_size=args.batch_size,
        phase=args.phase,
        book_weight=args.book_weight,
        num_workers=args.num_workers
    )
    
    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")
    
    # Create or load model
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        model = QuantumLLM.load(args.checkpoint, device)
    else:
        print(f"\nCreating new model...")
        model = QuantumLLM(DEFAULT_CONFIG)
    
    model = model.to(device)
    
    # Save config
    save_config(DEFAULT_CONFIG, output_dir / 'config.json')
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    scaler = GradScaler()
    
    # Calculate steps
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    print(f"\nTraining config:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Warmup steps: {warmup_steps:,}")
    print(f"  Max LR: {args.max_lr}")
    print(f"  Min LR: {args.min_lr}")
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    train_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scaler, device,
            epoch, args.max_lr, args.min_lr, warmup_steps, total_steps, global_step
        )
        
        # Evaluate
        val_loss, val_ppl = evaluate(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch}/{args.epochs} complete:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Perplexity: {val_ppl:.2f}")
        print(f"  Time: {epoch_time/60:.1f} min")
        
        # Save checkpoint
        checkpoint_path = output_dir / f'phase{args.phase}_epoch{epoch}.pt'
        model.save(checkpoint_path)
        print(f"  Saved: {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / f'phase{args.phase}_best.pt'
            model.save(best_path)
            print(f"  New best model saved!")
    
    total_time = time.time() - train_start
    
    # Save final model
    final_path = output_dir / f'phase{args.phase}_final.pt'
    model.save(final_path)
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} min")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val perplexity: {math.exp(best_val_loss):.2f}")
    print(f"Final model: {final_path}")
    
    # Test generation
    print(f"\n{'='*60}")
    print("Testing generation...")
    print(f"{'='*60}")
    
    model.eval()
    
    if args.phase == 1:
        # Book-style prompt
        prompts = [
            "Quantum computing is",
            "The qubit can exist in",
            "Entanglement allows two particles to"
        ]
    else:
        # Context-style prompt
        prompts = [
            "Context: Q: What is superposition? A: Superposition allows a qubit to be in multiple states at once. Question: What is a qubit? Answer:",
            "Context: Q: What is entanglement? A: Entanglement correlates two qubits. Question: How does quantum computing work? Answer:"
        ]
    
    for prompt in prompts:
        tokens = tokenizer.encode(prompt).ids
        x = torch.tensor([tokens], device=device)
        
        with torch.no_grad():
            output = model.generate(x, max_new_tokens=100, temperature=0.8)
        
        generated = tokenizer.decode(output[0].tolist())
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated[:300]}...")


if __name__ == "__main__":
    main()
