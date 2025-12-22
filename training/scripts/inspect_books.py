"""
Inspect Book Data Contribution

Only 3,830 book chunks vs 96,305 Q&A pairs.
Is the book data being drowned out?

Run on HPC.
"""

from pathlib import Path

BOOK_PATH = Path("data/combined_books.txt")
TOKENIZER_PATH = Path("tokenizer.json")
QA_COUNT = 96245


def main():
    print("=" * 60)
    print("BOOK DATA CONTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    
    # Load book data
    print(f"Loading {BOOK_PATH}...")
    with open(BOOK_PATH, 'r', encoding='utf-8') as f:
        book_text = f.read()
    
    # Tokenize
    print("Tokenizing book text...")
    book_tokens = tokenizer.encode(book_text).ids
    total_book_tokens = len(book_tokens)
    print(f"  Total book tokens: {total_book_tokens:,}")
    
    # Calculate chunks (from dataset.py: max_length=512, stride=256)
    max_length = 512
    stride = 256
    n_chunks = max(1, (total_book_tokens - max_length) // stride + 1)
    print(f"  Book chunks (stride={stride}): {n_chunks:,}")
    
    # Compare to Q&A
    print("\n" + "=" * 60)
    print("DATA BALANCE")
    print("=" * 60)
    
    total_examples = QA_COUNT + n_chunks
    qa_pct = QA_COUNT / total_examples * 100
    book_pct = n_chunks / total_examples * 100
    
    print(f"\n  Q&A pairs:    {QA_COUNT:,} ({qa_pct:.1f}%)")
    print(f"  Book chunks:  {n_chunks:,} ({book_pct:.1f}%)")
    print(f"  Total:        {total_examples:,}")
    
    print(f"\n  Ratio: {QA_COUNT / n_chunks:.1f}:1 (Q&A to book)")
    
    # Token contribution
    print("\n" + "=" * 60)
    print("TOKEN CONTRIBUTION (estimated)")
    print("=" * 60)
    
    # Estimate Q&A tokens (from earlier analysis: median 76 tokens)
    avg_qa_tokens = 76  # median from token analysis
    est_qa_tokens = QA_COUNT * avg_qa_tokens
    
    print(f"\n  Q&A tokens (est):  {est_qa_tokens:,} ({est_qa_tokens/(est_qa_tokens+total_book_tokens)*100:.1f}%)")
    print(f"  Book tokens:       {total_book_tokens:,} ({total_book_tokens/(est_qa_tokens+total_book_tokens)*100:.1f}%)")
    
    # Sample book content
    print("\n" + "=" * 60)
    print("BOOK CONTENT SAMPLE")
    print("=" * 60)
    
    # Split by book markers
    books = book_text.split("=" * 60)
    book_sections = [b.strip() for b in books if b.strip() and len(b.strip()) > 100]
    
    print(f"\n  Found {len(book_sections)} book sections")
    
    for i, section in enumerate(book_sections[:5], 1):
        lines = section.split('\n')
        title = lines[0] if lines else "Unknown"
        content_preview = ' '.join(section.split()[:50])
        print(f"\n  [{i}] {title[:50]}")
        print(f"      {content_preview}...")
    
    # Check book content quality
    print("\n" + "=" * 60)
    print("BOOK CONTENT ANALYSIS")
    print("=" * 60)
    
    # Check for common issues
    lines = book_text.split('\n')
    short_lines = sum(1 for l in lines if 0 < len(l.strip()) < 20)
    math_lines = sum(1 for l in lines if '$' in l or '\\' in l)
    
    print(f"\n  Total lines: {len(lines):,}")
    print(f"  Short lines (<20 chars): {short_lines:,} ({short_lines/len(lines)*100:.1f}%)")
    print(f"  Lines with math notation: {math_lines:,} ({math_lines/len(lines)*100:.1f}%)")
    
    # Training impact
    print("\n" + "=" * 60)
    print("TRAINING IMPACT")
    print("=" * 60)
    
    print(f"""
  Per epoch, model sees:
    - {QA_COUNT:,} Q&A examples
    - {n_chunks:,} book chunks
    
  Book data is {book_pct:.1f}% of training examples.
  
  With 3 epochs:
    - Each Q&A pair seen ~3 times
    - Each book chunk seen ~3 times
    
  Book contribution is small but consistent.
""")
    
    # Recommendation
    print("=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    if book_pct < 5:
        print(f"""
  ⚠️  Book data is only {book_pct:.1f}% of training data.
  
  Options:
    1. Upsample book data (repeat chunks)
    2. Downsample Q&A data (remove templates)
    3. Accept current ratio (Q&A format is primary use case)
    
  Given the Q&A templating issues found, Option 2 may naturally
  improve the balance after filtering templates.
""")
    else:
        print(f"\n  ✓ Book data contribution ({book_pct:.1f}%) is reasonable.")
    
    return 0


if __name__ == "__main__":
    exit(main())
