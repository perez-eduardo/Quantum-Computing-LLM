"""
Upsample Book Data 3x with Different Chunk Offsets

Creates 3 sets of book chunks with different starting offsets
to increase variety in book training data.

Offsets: 0, 85, 170 (stride=256, so ~1/3 stride apart)

Run on HPC:
    cd ~/hpc-share/quantum-llm
    source venv/bin/activate
    python scripts/upsample_books.py
"""

import json
from pathlib import Path


def main():
    # Paths (HPC)
    book_path = Path.home() / "hpc-share/quantum-llm/data/combined_books.txt"
    output_path = Path.home() / "hpc-share/quantum-llm/data/book_chunks_3x.jsonl"
    tokenizer_path = Path.home() / "hpc-share/quantum-llm/tokenizer.json"
    
    MAX_LENGTH = 512
    STRIDE = 256
    OFFSETS = [0, 85, 170]  # ~1/3 stride apart
    
    print("=" * 60)
    print("UPSAMPLE BOOKS 3x WITH DIFFERENT OFFSETS")
    print("=" * 60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    
    # Load book text
    print(f"\nLoading {book_path.name}...")
    with open(book_path, 'r', encoding='utf-8') as f:
        book_text = f.read()
    
    word_count = len(book_text.split())
    print(f"  Words: {word_count:,}")
    
    # Tokenize entire book
    print("\nTokenizing book text...")
    all_ids = tokenizer.encode(book_text).ids
    print(f"  Total tokens: {len(all_ids):,}")
    
    # Create chunks with different offsets
    print(f"\nCreating chunks (max_length={MAX_LENGTH}, stride={STRIDE})...")
    
    all_chunks = []
    
    for offset in OFFSETS:
        chunks_for_offset = []
        
        # Start from offset
        start = offset
        while start + MAX_LENGTH <= len(all_ids):
            chunk_ids = all_ids[start:start + MAX_LENGTH]
            chunks_for_offset.append({
                'ids': chunk_ids,
                'offset': offset,
                'start': start,
                'length': len(chunk_ids)
            })
            start += STRIDE
        
        # Handle last partial chunk if significant
        if start < len(all_ids) and len(all_ids) - start >= MAX_LENGTH // 2:
            chunk_ids = all_ids[start:]
            chunks_for_offset.append({
                'ids': chunk_ids,
                'offset': offset,
                'start': start,
                'length': len(chunk_ids)
            })
        
        print(f"  Offset {offset}: {len(chunks_for_offset):,} chunks")
        all_chunks.extend(chunks_for_offset)
    
    print(f"\n  Total chunks (3x): {len(all_chunks):,}")
    
    # Compare to original 1x
    original_chunks = (len(all_ids) - MAX_LENGTH) // STRIDE + 1
    print(f"  Original (1x): {original_chunks:,} chunks")
    print(f"  Multiplier: {len(all_chunks) / original_chunks:.1f}x")
    
    # Save as JSONL
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + '\n')
    
    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"  File size: {file_size:.1f} MB")
    
    # Verify
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    # Check chunk distribution by offset
    by_offset = {}
    for chunk in all_chunks:
        off = chunk['offset']
        by_offset[off] = by_offset.get(off, 0) + 1
    
    print("\nChunks by offset:")
    for off, count in sorted(by_offset.items()):
        print(f"  Offset {off}: {count:,}")
    
    # Sample chunks
    print("\nSample chunk token counts:")
    import random
    random.seed(42)
    samples = random.sample(all_chunks, min(5, len(all_chunks)))
    for i, chunk in enumerate(samples, 1):
        text_preview = tokenizer.decode(chunk['ids'][:20])
        print(f"  [{i}] Offset {chunk['offset']}, {chunk['length']} tokens: {text_preview}...")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nOutput: {output_path}")
    print("\nNote: dataset.py needs to be updated to load book_chunks_3x.jsonl")
    print("      instead of chunking on the fly.")
    
    return 0


if __name__ == "__main__":
    exit(main())
