"""
Chunk books into ~500 token pieces with overlap for RAG.
Run from project root: python backend/scripts/chunk_books.py

Output: data/processed/chunks.json
"""

import os
import json
from pathlib import Path


# Chunking parameters
CHUNK_SIZE = 500       # Target tokens per chunk
CHUNK_OVERLAP = 100    # Overlap between chunks
CHARS_PER_TOKEN = 4    # Rough estimate for character to token conversion


def estimate_tokens(text):
    """Estimate token count from character count."""
    return len(text) // CHARS_PER_TOKEN


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks.
    Uses character-based splitting with token estimation.
    """
    char_chunk_size = chunk_size * CHARS_PER_TOKEN
    char_overlap = overlap * CHARS_PER_TOKEN
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + char_chunk_size
        
        # Try to break at paragraph or sentence boundary
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind('\n\n', start + char_chunk_size // 2, end)
            if para_break != -1:
                end = para_break
            else:
                # Look for sentence break
                for punct in ['. ', '? ', '! ']:
                    sent_break = text.rfind(punct, start + char_chunk_size // 2, end)
                    if sent_break != -1:
                        end = sent_break + 1
                        break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start with overlap
        start = end - char_overlap
        if start < 0:
            start = 0
        
        # Prevent infinite loop
        if end >= len(text):
            break
    
    return chunks


def process_books(books_dir):
    """Process all book text files into chunks."""
    books_path = Path(books_dir)
    
    # Find all .txt files except combined.txt
    book_files = [f for f in books_path.glob("*.txt") if f.name != "combined.txt"]
    
    print(f"Found {len(book_files)} book files")
    
    all_chunks = []
    
    for book_file in book_files:
        book_name = book_file.stem
        print(f"\n  Processing: {book_name}")
        
        with open(book_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunks = chunk_text(text)
        
        print(f"    Characters: {len(text):,}")
        print(f"    Chunks: {len(chunks)}")
        
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'book_name': book_name,
                'chunk_index': i,
                'content': chunk,
                'est_tokens': estimate_tokens(chunk)
            })
    
    return all_chunks


def main():
    print("=" * 40)
    print("CHUNK BOOKS FOR RAG")
    print("=" * 40)
    print(f"\nChunk size: ~{CHUNK_SIZE} tokens")
    print(f"Overlap: ~{CHUNK_OVERLAP} tokens")
    
    # Paths
    books_dir = Path("data/raw/books")
    output_dir = Path("data/processed")
    output_file = output_dir / "chunks.json"
    
    # Check books directory
    if not books_dir.exists():
        print(f"\nERROR: Books directory not found: {books_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process books
    print(f"\nProcessing books from: {books_dir}")
    chunks = process_books(books_dir)
    
    # Save chunks
    print(f"\nSaving {len(chunks)} chunks to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2)
    
    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    
    # Stats by book
    book_counts = {}
    for chunk in chunks:
        book = chunk['book_name']
        book_counts[book] = book_counts.get(book, 0) + 1
    
    print("\nChunks per book:")
    for book, count in sorted(book_counts.items()):
        print(f"  {book}: {count}")
    
    print(f"\nTotal chunks: {len(chunks)}")
    
    # Token stats
    token_counts = [c['est_tokens'] for c in chunks]
    print(f"Tokens per chunk: min={min(token_counts)}, max={max(token_counts)}, avg={sum(token_counts)//len(token_counts)}")
    
    print(f"\nOutput: {output_file}")


if __name__ == "__main__":
    main()
