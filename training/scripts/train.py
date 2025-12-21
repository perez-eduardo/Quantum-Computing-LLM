"""
Quantum Computing LLM - Training Script
Trains the 1.2M parameter transformer on Q&A + book data
"""

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from model import create_model, QuantumLLM
from dataset import create_dataloaders, Tokenizer


def get_lr(step, warmup_steps, max_lr, min_lr, total_steps):
    """Cosine learning rate schedule with warmup"""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train_epoch(
    model, train_loader, optimizer, scaler, device,
    step, total_steps, warmup_steps, max_lr, min_lr, grad_clip,
    log_interval=100, writer=None
):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    tokens_processed = 0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        
        # Update learning rate
        lr = get_lr(step, warmup_steps, max_lr, min_lr, total_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass with mixed precision
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(batch, labels=batch)
            loss = outputs['loss']
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Stats
        total_loss += loss.item()
        tokens_processed += batch.numel()
        step += 1
        
        # Logging
        if step % log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed
            avg_loss = total_loss / (batch_idx + 1)
            
            print(f"Step {step:6d} | Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | "
                  f"LR: {lr:.2e} | Tok/s: {tokens_per_sec:.0f}")
            
            if writer:
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/avg_loss', avg_loss, step)
                writer.add_scalar('train/lr', lr, step)
                writer.add_scalar('train/tokens_per_sec', tokens_per_sec, step)
        
        if step >= total_steps:
            break
    
    return step, total_loss / len(train_loader)


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for batch in val_loader:
        batch = batch.to(device)
        
        with autocast():
            outputs = model(batch, labels=batch)
            loss = outputs['loss']
        
        # Count non-padding tokens
        non_pad = (batch != 0).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity


def save_checkpoint(model, optimizer, scaler, step, loss, path):
    """Save training checkpoint"""
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
    }, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(model, optimizer, scaler, path):
    """Load training checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return checkpoint['step'], checkpoint['loss']


def main(args):
    print("=" * 60)
    print("Quantum Computing LLM Training")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(output_dir / 'logs')
    
    # Create dataloaders
    print("\n" + "-" * 40)
    print("Loading data...")
    print("-" * 40)
    
    train_loader, val_loader, tokenizer = create_dataloaders(
        qa_path=args.qa_path,
        book_path=args.book_path,
        tokenizer_path=args.tokenizer_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )
    
    # Create model
    print("\n" + "-" * 40)
    print("Creating model...")
    print("-" * 40)
    
    model_config = {
        'vocab_size': len(tokenizer),
        'dim': args.dim,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'max_seq_len': args.max_length,
        'ff_dim': args.ff_dim,
        'dropout': args.dropout,
        'pad_token_id': tokenizer.pad_token_id,
    }
    
    model = create_model(model_config)
    model = model.to(device)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Calculate training steps
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    print(f"\nTraining config:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Warmup steps: {warmup_steps:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max LR: {args.max_lr}")
    print(f"  Min LR: {args.min_lr}")
    
    # Resume from checkpoint
    start_step = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_step, _ = load_checkpoint(model, optimizer, scaler, args.resume)
        print(f"Resumed at step {start_step}")
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    step = start_step
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---\n")
        
        step, train_loss = train_epoch(
            model, train_loader, optimizer, scaler, device,
            step, total_steps, warmup_steps,
            args.max_lr, args.min_lr, args.grad_clip,
            log_interval=args.log_interval,
            writer=writer,
        )
        
        # Evaluate
        val_loss, perplexity = evaluate(model, val_loader, device)
        print(f"\nEpoch {epoch + 1} complete:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        
        writer.add_scalar('val/loss', val_loss, step)
        writer.add_scalar('val/perplexity', perplexity, step)
        
        # Save checkpoint
        save_checkpoint(
            model, optimizer, scaler, step, val_loss,
            output_dir / f'checkpoint_epoch{epoch + 1}.pt'
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scaler, step, val_loss,
                output_dir / 'best_model.pt'
            )
            print(f"  New best model! Val Loss: {val_loss:.4f}")
        
        if step >= total_steps:
            break
    
    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pt')
    print(f"\nTraining complete! Final model saved to {output_dir / 'final_model.pt'}")
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Quantum Computing LLM")
    
    # Data paths
    parser.add_argument('--qa_path', type=str, default='../data/combined_qa_final.csv',
                        help='Path to Q&A CSV file')
    parser.add_argument('--book_path', type=str, default='../data/combined_books.txt',
                        help='Path to book text file')
    parser.add_argument('--tokenizer_path', type=str, default='../tokenizer.json',
                        help='Path to tokenizer JSON file')
    parser.add_argument('--output_dir', type=str, default='../model',
                        help='Output directory for checkpoints')
    
    # Model config
    parser.add_argument('--dim', type=int, default=64,
                        help='Model embedding dimension')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--ff_dim', type=int, default=256,
                        help='Feed-forward hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    
    # Training config
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--max_lr', type=float, default=3e-4,
                        help='Maximum learning rate')
    parser.add_argument('--min_lr', type=float, default=3e-5,
                        help='Minimum learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio of total steps')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--val_split', type=float, default=0.05,
                        help='Validation split ratio')
    
    # Other
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval (steps)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    main(args)
