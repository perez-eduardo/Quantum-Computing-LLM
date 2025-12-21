"""
Clean ChatGPT Q&A CSV

Removes template prefixes like "The term X refers to this:" from answers.

Usage:
    python clean_chatgpt_qa.py --input quantum_computing_qa_improved_v2.csv --output chatgpt_qa_cleaned.csv

Requirements:
    pip install pandas
"""

import argparse
import re
import pandas as pd
from pathlib import Path


def clean_answer(answer: str) -> str:
    """Remove template prefixes from answer."""
    if pd.isna(answer):
        return answer
    
    # Pattern: "The term X refers to this: <actual answer>"
    pattern = r'^The term\s+.+?\s+refers to this:\s*'
    cleaned = re.sub(pattern, '', str(answer), flags=re.IGNORECASE)
    
    return cleaned.strip()


def main():
    parser = argparse.ArgumentParser(description="Clean ChatGPT Q&A CSV")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: {args.input} not found")
        return 1
    
    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    
    print(f"Rows: {len(df):,}")
    
    # Count rows with template prefix before cleaning
    pattern_count = df.iloc[:,1].str.contains(r'^The term\s+.+?\s+refers to this:', case=False, na=False, regex=True).sum()
    print(f"Rows with template prefix: {pattern_count:,}")
    
    # Clean answers
    print("Cleaning answers...")
    df.iloc[:,1] = df.iloc[:,1].apply(clean_answer)
    
    # Verify
    pattern_count_after = df.iloc[:,1].str.contains(r'^The term\s+.+?\s+refers to this:', case=False, na=False, regex=True).sum()
    print(f"Rows with template prefix after: {pattern_count_after:,}")
    
    # Save
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")
    
    # Show samples
    print("\n--- SAMPLE CLEANED ROWS ---")
    for i in [0, 1, 2]:
        print(f"Q: {df.iloc[i,0][:80]}")
        print(f"A: {df.iloc[i,1][:120]}...")
        print()
    
    return 0


if __name__ == "__main__":
    exit(main())
