"""
Extract Text from Book PDFs

Extracts and cleans text from quantum computing book PDFs.

Usage:
    python extract_books.py --input-dir data/raw/books/pdf --output-dir data/raw/books

Requirements:
    pip install pymupdf

Note: Place your PDF files in the input directory before running.
"""

import argparse
import re
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: pymupdf not installed. Run: pip install pymupdf")
    exit(1)


# Map PDF filenames to clean output names
BOOK_NAMES = {
    "Quantum Computing Explained for Beginners": "beginners",
    "Quantum Computing for Everyone": "bernhardt",
    "Introduction to Classical and Quantum Computing": "wong",
    "Quantum Computation and Quantum Information": "nielsen_chuang",
    "Quantum Computing An Applied Approach": "hidary",
}


def get_output_name(pdf_name: str) -> str:
    """Map PDF filename to clean output name."""
    for key, value in BOOK_NAMES.items():
        if key.lower() in pdf_name.lower():
            return value
    # Fallback: sanitize filename
    return re.sub(r'[^\w]+', '_', pdf_name.lower()).strip('_')


def clean_text(text: str) -> str:
    """Clean extracted text."""
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Normalize whitespace (but preserve paragraph breaks)
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n{3,}', '\n\n', text)  # 3+ newlines to 2
    
    # Remove page numbers (common patterns)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Standalone numbers on lines
    
    # Strip lines that are just whitespace
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()


def extract_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    
    text_parts = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            text_parts.append(text)
    
    doc.close()
    
    full_text = '\n'.join(text_parts)
    return clean_text(full_text)


def main():
    parser = argparse.ArgumentParser(description="Extract text from book PDFs")
    parser.add_argument("--input-dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output-dir", required=True, help="Directory for output text files")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found in {input_dir}")
        return 1
    
    print(f"Found {len(pdf_files)} PDF files\n")
    
    all_texts = []
    
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        
        # Extract text
        text = extract_pdf(pdf_path)
        
        # Get output name
        output_name = get_output_name(pdf_path.stem)
        output_path = output_dir / f"{output_name}.txt"
        
        # Save individual file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Stats
        word_count = len(text.split())
        char_count = len(text)
        print(f"  -> {output_path.name}: {word_count:,} words, {char_count:,} chars")
        
        # Collect for combined file
        header = f"{'='*60}\n{output_name.upper()}\n{'='*60}\n\n"
        all_texts.append(header + text)
    
    # Save combined file
    combined_path = output_dir / "combined_books.txt"
    with open(combined_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_texts))
    
    combined_words = len(open(combined_path, encoding='utf-8').read().split())
    print(f"\nCombined: {combined_path}")
    print(f"  Total words: {combined_words:,}")
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
