"""
Inspect Book Text File

Detailed inspection of a single book text file for quality issues.

Usage:
    python inspect_book.py --input data/raw/books/wong.txt
"""

import argparse
import re
from pathlib import Path
from collections import Counter


def check_control_chars(text: str) -> list[str]:
    """Find control characters."""
    found = []
    for i, char in enumerate(text[:10000]):  # Sample first 10k chars
        if ord(char) < 32 and char not in '\n\r\t':
            found.append(f"Position {i}: ord={ord(char)}")
    return found[:10]


def find_garbage_lines(lines: list[str]) -> list[tuple[int, str]]:
    """Find lines that look like garbage."""
    garbage = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        
        # Very short non-empty lines (likely fragments)
        if 0 < len(stripped) < 5:
            garbage.append((i+1, f"Very short: '{stripped}'"))
            continue
        
        # Lines that are mostly non-alphabetic
        alpha_ratio = sum(c.isalpha() for c in stripped) / max(len(stripped), 1)
        if len(stripped) > 10 and alpha_ratio < 0.3:
            garbage.append((i+1, f"Low alpha ({alpha_ratio:.1%}): '{stripped[:50]}'"))
            continue
        
        # Lines that look like page numbers or headers
        if re.match(r'^\d+$', stripped):  # Just a number
            garbage.append((i+1, f"Standalone number: '{stripped}'"))
            continue
    
    return garbage[:30]


def analyze_paragraph_lengths(text: str) -> dict:
    """Analyze paragraph (double-newline separated) lengths."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    if not paragraphs:
        return {"count": 0}
    
    lengths = [len(p) for p in paragraphs]
    
    return {
        "count": len(paragraphs),
        "min": min(lengths),
        "max": max(lengths),
        "median": sorted(lengths)[len(lengths)//2],
        "very_short": sum(1 for l in lengths if l < 20),
        "short": sum(1 for l in lengths if 20 <= l < 50),
        "medium": sum(1 for l in lengths if 50 <= l < 200),
        "long": sum(1 for l in lengths if l >= 200),
    }


def sample_short_paragraphs(text: str, threshold: int = 30, n: int = 15) -> list[str]:
    """Get samples of short paragraphs."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    short = [p for p in paragraphs if 0 < len(p) < threshold]
    return short[:n]


def sample_random_paragraphs(text: str, n: int = 5) -> list[str]:
    """Get random paragraph samples."""
    import random
    random.seed(42)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p) > 50]
    if len(paragraphs) <= n:
        return paragraphs
    return random.sample(paragraphs, n)


def main():
    parser = argparse.ArgumentParser(description="Inspect book text file")
    parser.add_argument("--input", required=True, help="Path to book .txt file")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: {args.input} not found")
        return 1
    
    filename = Path(args.input).name
    
    print("=" * 60)
    print(f"BOOK INSPECTION: {filename}")
    print("=" * 60)
    
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()
    
    lines = text.split('\n')
    
    # Basic stats
    print(f"\nTotal characters: {len(text):,}")
    print(f"Total lines: {len(lines):,}")
    print(f"Total words: {len(text.split()):,}")
    
    # Paragraph analysis
    print("\n--- PARAGRAPH ANALYSIS ---")
    para_stats = analyze_paragraph_lengths(text)
    print(f"Total paragraphs: {para_stats['count']:,}")
    print(f"Length range: {para_stats['min']} - {para_stats['max']} chars")
    print(f"Median length: {para_stats['median']} chars")
    print(f"Very short (<20 chars): {para_stats['very_short']}")
    print(f"Short (20-50 chars): {para_stats['short']}")
    print(f"Medium (50-200 chars): {para_stats['medium']}")
    print(f"Long (200+ chars): {para_stats['long']}")
    
    # Control characters
    print("\n--- CONTROL CHARACTERS ---")
    control = check_control_chars(text)
    if control:
        print(f"Found {len(control)} control characters (showing first 10):")
        for c in control:
            print(f"  {c}")
    else:
        print("None found")
    
    # Garbage lines
    print("\n--- POTENTIAL GARBAGE LINES ---")
    garbage = find_garbage_lines(lines)
    if garbage:
        print(f"Found {len(garbage)} suspicious lines (showing first 30):")
        for line_num, desc in garbage:
            print(f"  Line {line_num}: {desc}")
    else:
        print("None found")
    
    # Short paragraph samples
    print("\n--- SHORT PARAGRAPH SAMPLES (<30 chars) ---")
    short_paras = sample_short_paragraphs(text, threshold=30, n=15)
    if short_paras:
        for i, p in enumerate(short_paras, 1):
            display = p.replace('\n', ' ')[:60]
            print(f"  {i}. '{display}'")
    else:
        print("None found")
    
    # Random good paragraph samples
    print("\n--- RANDOM PARAGRAPH SAMPLES ---")
    samples = sample_random_paragraphs(text, n=3)
    for i, p in enumerate(samples, 1):
        display = p.replace('\n', ' ')[:200]
        print(f"\n[Sample {i}]")
        print(f"{display}...")
    
    # First 500 chars
    print("\n--- FIRST 500 CHARACTERS ---")
    print(text[:500])
    
    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
