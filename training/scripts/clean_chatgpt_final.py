"""
Clean ChatGPT Q&A Data (Final Pass)

Removes:
- Very short questions (<20 chars)
- Very short answers (<50 chars)
- Duplicate answers (keeps first occurrence)

Usage:
    python clean_chatgpt_final.py --input data/raw/chatgpt_qa_cleaned.csv --output data/raw/chatgpt_qa_final.csv
"""

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Clean ChatGPT Q&A (final pass)")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--min-question-len", type=int, default=20, help="Minimum question length")
    parser.add_argument("--min-answer-len", type=int, default=50, help="Minimum answer length")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: {args.input} not found")
        return 1
    
    print("=" * 60)
    print("CLEANING CHATGPT DATA (FINAL PASS)")
    print("=" * 60)
    
    df = pd.read_csv(args.input)
    print(f"\nInput rows: {len(df):,}")
    
    # Detect column names
    q_col = df.columns[0]
    a_col = df.columns[1]
    print(f"Question column: '{q_col}'")
    print(f"Answer column: '{a_col}'")
    
    # Step 1: Remove short questions
    print(f"\n[1] Removing questions shorter than {args.min_question_len} chars...")
    df['q_len'] = df[q_col].astype(str).str.len()
    short_q = df[df['q_len'] < args.min_question_len]
    
    if len(short_q) > 0:
        print(f"    Found {len(short_q)} short questions:")
        for _, row in short_q.head(10).iterrows():
            print(f"    - '{row[q_col]}'")
        if len(short_q) > 10:
            print(f"    ... and {len(short_q) - 10} more")
    
    df = df[df['q_len'] >= args.min_question_len]
    print(f"    Removed: {len(short_q)} rows")
    
    # Step 2: Remove short answers
    print(f"\n[2] Removing answers shorter than {args.min_answer_len} chars...")
    df['a_len'] = df[a_col].astype(str).str.len()
    short_a = df[df['a_len'] < args.min_answer_len]
    
    if len(short_a) > 0:
        print(f"    Found {len(short_a)} short answers:")
        for _, row in short_a.head(10).iterrows():
            print(f"    - Q: '{row[q_col][:50]}...'")
            print(f"      A: '{row[a_col]}'")
        if len(short_a) > 10:
            print(f"    ... and {len(short_a) - 10} more")
    
    df = df[df['a_len'] >= args.min_answer_len]
    print(f"    Removed: {len(short_a)} rows")
    
    # Step 3: Remove duplicate answers
    print("\n[3] Removing duplicate answers...")
    dupes_before = df[a_col].duplicated().sum()
    df = df.drop_duplicates(subset=[a_col], keep='first')
    print(f"    Removed: {dupes_before:,} duplicate answers")
    
    # Step 4: Remove duplicate questions (if any)
    print("\n[4] Removing duplicate questions...")
    dupes_q = df[q_col].duplicated().sum()
    df = df.drop_duplicates(subset=[q_col], keep='first')
    print(f"    Removed: {dupes_q} duplicate questions")
    
    # Drop helper columns
    df = df.drop(columns=['q_len', 'a_len'])
    
    # Save
    df.to_csv(args.output, index=False)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Input rows:  {len(pd.read_csv(args.input)):,}")
    print(f"Output rows: {len(df):,}")
    print(f"Removed:     {len(pd.read_csv(args.input)) - len(df):,} rows total")
    print(f"Saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
