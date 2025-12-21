"""
Clean Stack Exchange Q&A Data

Fixes:
- HTML entities (&amp; -> &, &bull; -> •, etc.)
- Removes very short answers (<50 chars)
- Removes duplicate answers

Usage:
    python clean_stackexchange.py --input data/raw/stackexchange_qa.csv --output data/raw/stackexchange_qa_cleaned.csv
"""

import argparse
import html
import re
from pathlib import Path
import pandas as pd


def decode_html_entities(text: str) -> str:
    """Decode HTML entities to their characters."""
    if pd.isna(text):
        return text
    return html.unescape(str(text))


def main():
    parser = argparse.ArgumentParser(description="Clean Stack Exchange Q&A")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--min-answer-len", type=int, default=50, help="Minimum answer length in chars")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: {args.input} not found")
        return 1
    
    print("=" * 60)
    print("CLEANING STACK EXCHANGE DATA")
    print("=" * 60)
    
    df = pd.read_csv(args.input)
    print(f"\nInput rows: {len(df):,}")
    
    # Step 1: Decode HTML entities
    print("\n[1] Decoding HTML entities...")
    entities_before = df['question'].str.contains(r'&[a-zA-Z]+;|&#\d+;', regex=True, na=False).sum()
    entities_before += df['answer'].str.contains(r'&[a-zA-Z]+;|&#\d+;', regex=True, na=False).sum()
    
    df['question'] = df['question'].apply(decode_html_entities)
    df['answer'] = df['answer'].apply(decode_html_entities)
    
    entities_after = df['question'].str.contains(r'&[a-zA-Z]+;|&#\d+;', regex=True, na=False).sum()
    entities_after += df['answer'].str.contains(r'&[a-zA-Z]+;|&#\d+;', regex=True, na=False).sum()
    print(f"    HTML entities: {entities_before} -> {entities_after}")
    
    # Step 2: Remove very short answers
    print(f"\n[2] Removing answers shorter than {args.min_answer_len} chars...")
    df['a_len'] = df['answer'].astype(str).str.len()
    short_answers = df[df['a_len'] < args.min_answer_len]
    
    if len(short_answers) > 0:
        print(f"    Found {len(short_answers)} short answers:")
        for _, row in short_answers.iterrows():
            print(f"    - Q: {str(row['question'])[:60]}...")
            print(f"      A: {str(row['answer'])}")
            print()
    
    df = df[df['a_len'] >= args.min_answer_len]
    print(f"    Removed: {len(short_answers)} rows")
    
    # Step 3: Remove duplicate answers
    print("\n[3] Removing duplicate answers...")
    dupes_before = df['answer'].duplicated().sum()
    df = df.drop_duplicates(subset=['answer'], keep='first')
    print(f"    Removed: {dupes_before} duplicate answers")
    
    # Step 4: Remove duplicate questions (if any)
    print("\n[4] Removing duplicate questions...")
    dupes_q = df['question'].duplicated().sum()
    df = df.drop_duplicates(subset=['question'], keep='first')
    print(f"    Removed: {dupes_q} duplicate questions")
    
    # Drop helper column
    df = df.drop(columns=['a_len'])
    
    # Save
    df.to_csv(args.output, index=False)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Input rows:  {len(pd.read_csv(args.input)):,}")
    print(f"Output rows: {len(df):,}")
    print(f"Saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
