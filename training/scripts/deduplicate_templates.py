"""
Deduplicate Templated Questions

Removes duplicate templated questions (e.g., same question with different numbers).
Keeps 1 copy of each template pattern.

Run on HPC:
    cd ~/hpc-share/quantum-llm
    source venv/bin/activate
    python scripts/deduplicate_templates.py
"""

import pandas as pd
import re
from pathlib import Path


def normalize_question(q):
    """Replace numbers with placeholder to find templates."""
    normalized = re.sub(r'\b\d+\.?\d*\b', '<NUM>', str(q))
    return normalized.strip()


def main():
    # Paths (HPC)
    input_path = Path.home() / "hpc-share/quantum-llm/data/chatgpt_qa_cleaned_v2.csv"
    output_path = Path.home() / "hpc-share/quantum-llm/data/chatgpt_qa_deduped.csv"
    
    print("=" * 60)
    print("DEDUPLICATE TEMPLATED QUESTIONS")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  Input rows: {len(df):,}")
    
    # Detect column names
    q_col = df.columns[0]
    a_col = df.columns[1]
    print(f"  Question column: '{q_col}'")
    print(f"  Answer column: '{a_col}'")
    
    # Normalize questions
    print("\nNormalizing questions...")
    df['normalized'] = df[q_col].apply(normalize_question)
    
    # Count before
    unique_before = df['normalized'].nunique()
    print(f"  Unique patterns: {unique_before:,}")
    
    # Keep first occurrence of each normalized pattern
    print("\nDeduplicating (keeping first of each pattern)...")
    df_deduped = df.drop_duplicates(subset=['normalized'], keep='first')
    
    # Drop the normalized column
    df_deduped = df_deduped.drop(columns=['normalized'])
    
    removed = len(df) - len(df_deduped)
    print(f"  Removed: {removed:,} rows ({removed/len(df)*100:.1f}%)")
    print(f"  Remaining: {len(df_deduped):,} rows")
    
    # Save
    print(f"\nSaving to {output_path}...")
    df_deduped.to_csv(output_path, index=False)
    
    # Verify
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    df_check = pd.read_csv(output_path)
    df_check['normalized'] = df_check[q_col].apply(normalize_question)
    
    # Check for remaining duplicates
    dup_count = df_check['normalized'].duplicated().sum()
    print(f"  Duplicate patterns remaining: {dup_count}")
    
    # Sample output
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT (5 random rows)")
    print("=" * 60)
    
    samples = df_deduped.sample(5, random_state=42)
    for i, (_, row) in enumerate(samples.iterrows(), 1):
        print(f"\n[{i}]")
        print(f"  Q: {row[q_col][:80]}...")
        print(f"  A: {row[a_col][:100]}...")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nOutput: {output_path}")
    print(f"Next: Combine with Stack Exchange and rebuild combined_qa_final_v2.csv")
    
    return 0


if __name__ == "__main__":
    exit(main())
