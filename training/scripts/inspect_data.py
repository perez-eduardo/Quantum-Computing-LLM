"""
Data Quality Inspection Script

Checks combined_qa.csv and combined_books.txt for potential issues.
Run locally before training.

Usage:
    python inspect_data.py --qa data/raw/combined_qa.csv --books data/raw/combined_books.txt

Requirements:
    pip install pandas
"""

import argparse
import re
from pathlib import Path
from collections import Counter
import pandas as pd


def check_html_artifacts(text: str) -> list[str]:
    """Find HTML tags or entities in text."""
    issues = []
    
    # HTML tags
    tags = re.findall(r'<[^>]+>', text)
    if tags:
        issues.append(f"HTML tags: {tags[:5]}")
    
    # HTML entities
    entities = re.findall(r'&[a-zA-Z]+;|&#\d+;', text)
    if entities:
        issues.append(f"HTML entities: {entities[:5]}")
    
    return issues


def check_encoding_issues(text: str) -> list[str]:
    """Find potential encoding problems."""
    issues = []
    
    # Replacement characters
    if '\ufffd' in text:
        issues.append("Contains replacement character (encoding error)")
    
    # Suspicious patterns
    if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', text):
        issues.append("Contains control characters")
    
    return issues


def inspect_qa_data(qa_path: str):
    """Inspect Q&A CSV file."""
    print("=" * 60)
    print("Q&A DATA INSPECTION")
    print("=" * 60)
    
    df = pd.read_csv(qa_path)
    
    # Basic stats
    print(f"\nRows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for required columns
    if 'question' not in df.columns or 'answer' not in df.columns:
        print("\nERROR: Missing 'question' or 'answer' column!")
        return
    
    # Null/empty checks
    print("\n--- NULL/EMPTY CHECK ---")
    null_q = df['question'].isna().sum()
    null_a = df['answer'].isna().sum()
    empty_q = (df['question'].astype(str).str.strip() == '').sum()
    empty_a = (df['answer'].astype(str).str.strip() == '').sum()
    
    print(f"Null questions: {null_q}")
    print(f"Null answers: {null_a}")
    print(f"Empty questions: {empty_q}")
    print(f"Empty answers: {empty_a}")
    
    # Length stats
    print("\n--- LENGTH STATS (characters) ---")
    df['q_len'] = df['question'].astype(str).str.len()
    df['a_len'] = df['answer'].astype(str).str.len()
    
    print(f"Question length: min={df['q_len'].min()}, median={df['q_len'].median():.0f}, max={df['q_len'].max()}")
    print(f"Answer length: min={df['a_len'].min()}, median={df['a_len'].median():.0f}, max={df['a_len'].max()}")
    
    # Very short entries
    short_q = (df['q_len'] < 10).sum()
    short_a = (df['a_len'] < 20).sum()
    print(f"\nVery short questions (<10 chars): {short_q}")
    print(f"Very short answers (<20 chars): {short_a}")
    
    # Very long entries
    long_q = (df['q_len'] > 1000).sum()
    long_a = (df['a_len'] > 5000).sum()
    print(f"Very long questions (>1000 chars): {long_q}")
    print(f"Very long answers (>5000 chars): {long_a}")
    
    # Duplicates
    print("\n--- DUPLICATES ---")
    dup_q = df['question'].duplicated().sum()
    dup_a = df['answer'].duplicated().sum()
    dup_both = df.duplicated(subset=['question', 'answer']).sum()
    print(f"Duplicate questions: {dup_q}")
    print(f"Duplicate answers: {dup_a}")
    print(f"Duplicate Q&A pairs: {dup_both}")
    
    # Source distribution (if exists)
    if 'source' in df.columns:
        print("\n--- SOURCE DISTRIBUTION ---")
        print(df['source'].value_counts())
    
    # HTML/encoding issues (sample check)
    print("\n--- HTML/ENCODING ISSUES (sample of 1000) ---")
    sample = df.sample(min(1000, len(df)), random_state=42)
    html_count = 0
    encoding_count = 0
    
    for _, row in sample.iterrows():
        text = str(row['question']) + " " + str(row['answer'])
        if check_html_artifacts(text):
            html_count += 1
        if check_encoding_issues(text):
            encoding_count += 1
    
    print(f"Rows with HTML artifacts: ~{html_count} in sample ({html_count/10:.1f}% estimated)")
    print(f"Rows with encoding issues: ~{encoding_count} in sample ({encoding_count/10:.1f}% estimated)")
    
    # Show examples with issues
    print("\n--- SAMPLE ENTRIES WITH HTML (up to 3) ---")
    shown = 0
    for _, row in df.iterrows():
        text = str(row['question']) + " " + str(row['answer'])
        issues = check_html_artifacts(text)
        if issues:
            print(f"\nQ: {str(row['question'])[:100]}...")
            print(f"A: {str(row['answer'])[:100]}...")
            print(f"Issues: {issues}")
            shown += 1
            if shown >= 3:
                break
    
    if shown == 0:
        print("None found!")
    
    # Random samples
    print("\n--- RANDOM SAMPLES (3) ---")
    samples = df.sample(3, random_state=123)
    for i, (_, row) in enumerate(samples.iterrows(), 1):
        print(f"\n[Sample {i}]")
        print(f"Q: {str(row['question'])[:200]}")
        print(f"A: {str(row['answer'])[:200]}...")


def inspect_book_data(book_path: str):
    """Inspect book text file."""
    print("\n" + "=" * 60)
    print("BOOK DATA INSPECTION")
    print("=" * 60)
    
    with open(book_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Basic stats
    print(f"\nTotal characters: {len(text):,}")
    print(f"Total words (approx): {len(text.split()):,}")
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    print(f"Paragraphs: {len(paragraphs):,}")
    
    # Paragraph length stats
    para_lens = [len(p) for p in paragraphs]
    print(f"\nParagraph length: min={min(para_lens)}, median={sorted(para_lens)[len(para_lens)//2]}, max={max(para_lens)}")
    
    # Very short paragraphs
    short_paras = sum(1 for p in para_lens if p < 50)
    print(f"Very short paragraphs (<50 chars): {short_paras}")
    
    # HTML/encoding check
    print("\n--- HTML/ENCODING ISSUES ---")
    html_issues = check_html_artifacts(text)
    encoding_issues = check_encoding_issues(text)
    
    if html_issues:
        print(f"HTML issues: {html_issues}")
    else:
        print("No HTML artifacts found")
    
    if encoding_issues:
        print(f"Encoding issues: {encoding_issues}")
    else:
        print("No encoding issues found")
    
    # Sample paragraphs
    print("\n--- SAMPLE PARAGRAPHS (3) ---")
    import random
    random.seed(42)
    for i, para in enumerate(random.sample(paragraphs, min(3, len(paragraphs))), 1):
        print(f"\n[Sample {i}]")
        print(para[:300] + "..." if len(para) > 300 else para)


def main():
    parser = argparse.ArgumentParser(description="Inspect data quality")
    parser.add_argument("--qa", required=True, help="Path to combined_qa.csv")
    parser.add_argument("--books", required=True, help="Path to combined_books.txt")
    
    args = parser.parse_args()
    
    if not Path(args.qa).exists():
        print(f"Error: {args.qa} not found")
        return 1
    if not Path(args.books).exists():
        print(f"Error: {args.books} not found")
        return 1
    
    inspect_qa_data(args.qa)
    inspect_book_data(args.books)
    
    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
