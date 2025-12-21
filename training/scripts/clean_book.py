"""
Clean Book Text Files

Removes garbage from PDF-extracted book text:
- Very short paragraphs (math fragments)
- Table of contents lines (dots)
- Roman numeral page numbers
- URLs and ISBNs
- Low-alpha lines

Usage:
    python clean_book.py --input data/raw/books/wong.txt --output data/raw/books/wong_cleaned.txt

    Or clean all books:
    python clean_book.py --input-dir data/raw/books --output-dir data/raw/books/cleaned
"""

import argparse
import re
from pathlib import Path


def is_roman_numeral(text: str) -> bool:
    """Check if text is a Roman numeral."""
    text = text.strip().lower()
    return bool(re.match(r'^[ivxlcdm]+$', text))


def is_toc_line(text: str) -> bool:
    """Check if line looks like table of contents (many dots)."""
    dot_count = text.count('.') + text.count('. ')
    if dot_count >= 5 and len(text) > 20:
        # Check if it's mostly dots and spaces
        dots_spaces = sum(1 for c in text if c in '. ')
        if dots_spaces / len(text) > 0.4:
            return True
    return False


def is_url_or_isbn(text: str) -> bool:
    """Check if line is URL or ISBN."""
    text_lower = text.lower()
    if 'http' in text_lower or 'https' in text_lower:
        return True
    if 'doi.org' in text_lower:
        return True
    if re.match(r'^isbn[:\s]', text_lower):
        return True
    if re.match(r'^[\d\-]+$', text.strip()) and len(text.strip()) > 10:
        return True
    return False


def is_garbage_line(line: str) -> bool:
    """Check if a line is garbage."""
    stripped = line.strip()
    
    # Empty
    if not stripped:
        return False  # Keep empty lines for paragraph breaks
    
    # Very short (likely math fragments)
    if len(stripped) < 5:
        return True
    
    # Roman numerals (page numbers)
    if is_roman_numeral(stripped):
        return True
    
    # Just a number (page number)
    if re.match(r'^\d+$', stripped):
        return True
    
    # Table of contents
    if is_toc_line(stripped):
        return True
    
    # URL or ISBN
    if is_url_or_isbn(stripped):
        return True
    
    # Lines that are mostly dashes or underscores
    if len(stripped) > 5:
        dash_count = sum(1 for c in stripped if c in '-_=')
        if dash_count / len(stripped) > 0.5:
            return True
    
    # Lines with very low alphabetic ratio (mostly symbols/numbers)
    if len(stripped) > 10:
        alpha_count = sum(1 for c in stripped if c.isalpha())
        if alpha_count / len(stripped) < 0.25:
            return True
    
    return False


def is_garbage_paragraph(para: str, min_length: int = 50) -> bool:
    """Check if a paragraph should be removed."""
    stripped = para.strip()
    
    # Too short
    if len(stripped) < min_length:
        return True
    
    # Very low alpha ratio for the whole paragraph
    alpha_count = sum(1 for c in stripped if c.isalpha())
    if len(stripped) > 20 and alpha_count / len(stripped) < 0.3:
        return True
    
    return False


def clean_book_text(text: str, min_para_length: int = 50) -> str:
    """Clean book text by removing garbage."""
    
    # Step 1: Clean individual lines
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if not is_garbage_line(line):
            cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # Step 2: Clean paragraphs
    paragraphs = text.split('\n\n')
    cleaned_paragraphs = []
    
    for para in paragraphs:
        if not is_garbage_paragraph(para, min_para_length):
            cleaned_paragraphs.append(para.strip())
    
    # Step 3: Reassemble with consistent paragraph breaks
    text = '\n\n'.join(p for p in cleaned_paragraphs if p)
    
    # Step 4: Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Single spaces
    
    return text.strip()


def process_single_file(input_path: Path, output_path: Path, min_para_length: int):
    """Process a single book file."""
    print(f"\nProcessing: {input_path.name}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    original_chars = len(text)
    original_words = len(text.split())
    
    cleaned = clean_book_text(text, min_para_length)
    
    cleaned_chars = len(cleaned)
    cleaned_words = len(cleaned.split())
    
    # Count paragraphs
    original_paras = len([p for p in text.split('\n\n') if p.strip()])
    cleaned_paras = len([p for p in cleaned.split('\n\n') if p.strip()])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)
    
    print(f"  Characters: {original_chars:,} -> {cleaned_chars:,} ({cleaned_chars/original_chars*100:.1f}%)")
    print(f"  Words: {original_words:,} -> {cleaned_words:,} ({cleaned_words/original_words*100:.1f}%)")
    print(f"  Paragraphs: {original_paras} -> {cleaned_paras}")
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Clean book text files")
    parser.add_argument("--input", help="Single input file path")
    parser.add_argument("--output", help="Single output file path")
    parser.add_argument("--input-dir", help="Input directory with .txt files")
    parser.add_argument("--output-dir", help="Output directory for cleaned files")
    parser.add_argument("--min-para-length", type=int, default=50, help="Minimum paragraph length to keep")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CLEANING BOOK TEXT FILES")
    print("=" * 60)
    print(f"Minimum paragraph length: {args.min_para_length} chars")
    
    if args.input and args.output:
        # Single file mode
        process_single_file(Path(args.input), Path(args.output), args.min_para_length)
    
    elif args.input_dir and args.output_dir:
        # Directory mode
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all .txt files except combined_books.txt
        txt_files = [f for f in input_dir.glob("*.txt") if f.name != "combined_books.txt"]
        
        if not txt_files:
            print(f"No .txt files found in {input_dir}")
            return 1
        
        print(f"\nFound {len(txt_files)} files to process")
        
        for txt_file in txt_files:
            output_path = output_dir / f"{txt_file.stem}_cleaned.txt"
            process_single_file(txt_file, output_path, args.min_para_length)
        
        # Create new combined file
        print("\n" + "-" * 40)
        print("Creating combined_books.txt...")
        
        all_texts = []
        for txt_file in sorted(output_dir.glob("*_cleaned.txt")):
            with open(txt_file, 'r', encoding='utf-8') as f:
                book_text = f.read()
            header = f"{'='*60}\n{txt_file.stem.upper().replace('_CLEANED', '')}\n{'='*60}\n\n"
            all_texts.append(header + book_text)
        
        combined_path = output_dir / "combined_books.txt"
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(all_texts))
        
        combined_words = len(open(combined_path, encoding='utf-8').read().split())
        print(f"  Combined file: {combined_path}")
        print(f"  Total words: {combined_words:,}")
    
    else:
        print("Error: Provide either --input/--output or --input-dir/--output-dir")
        return 1
    
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
