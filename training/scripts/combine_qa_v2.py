"""
Combine Q&A Sources v2

Merges Claude Q&A and Stack Exchange into combined_qa.csv.
ChatGPT data has been abandoned.

Run on HPC:
    cd ~/hpc-share/quantum-llm
    source venv/bin/activate
    python scripts/combine_qa_v2.py
"""

import pandas as pd
from pathlib import Path


def main():
    data_dir = Path.home() / "hpc-share/quantum-llm/data"
    
    claude_path = data_dir / "claude_qa.csv"
    stackexchange_path = data_dir / "stackexchange_qa_cleaned.csv"
    output_path = data_dir / "combined_qa.csv"
    
    print("=" * 60)
    print("COMBINE Q&A SOURCES v2")
    print("=" * 60)
    print("\nSources:")
    print("  - Claude Q&A (15,000 pairs)")
    print("  - Stack Exchange (cleaned)")
    print("  - ChatGPT: ABANDONED (94% garbage)")
    
    # Load Claude Q&A
    print(f"\n[1] Loading {claude_path.name}...")
    if not claude_path.exists():
        print(f"  ERROR: File not found: {claude_path}")
        return 1
    
    df_claude = pd.read_csv(claude_path)
    print(f"  Rows: {len(df_claude):,}")
    print(f"  Columns: {list(df_claude.columns)}")
    
    # Standardize columns
    df_claude = df_claude.rename(columns={
        df_claude.columns[0]: 'question',
        df_claude.columns[1]: 'answer'
    })
    df_claude['source'] = 'claude_synthetic'
    
    # Load Stack Exchange
    print(f"\n[2] Loading {stackexchange_path.name}...")
    if not stackexchange_path.exists():
        print(f"  ERROR: File not found: {stackexchange_path}")
        return 1
    
    df_se = pd.read_csv(stackexchange_path)
    print(f"  Rows: {len(df_se):,}")
    print(f"  Columns: {list(df_se.columns)}")
    
    # Standardize columns
    df_se = df_se.rename(columns={
        df_se.columns[0]: 'question',
        df_se.columns[1]: 'answer'
    })
    df_se['source'] = 'stackexchange'
    
    # Combine
    print("\n[3] Combining datasets...")
    df_combined = pd.concat([df_claude, df_se], ignore_index=True)
    print(f"  Combined rows: {len(df_combined):,}")
    
    # Basic cleaning
    print("\n[4] Basic cleaning...")
    
    # Remove empty
    before = len(df_combined)
    df_combined = df_combined.dropna(subset=['question', 'answer'])
    df_combined = df_combined[df_combined['question'].str.strip() != '']
    df_combined = df_combined[df_combined['answer'].str.strip() != '']
    after = len(df_combined)
    print(f"  Removed empty: {before - after}")
    
    # Remove exact duplicate Q&A pairs
    before = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['question', 'answer'])
    after = len(df_combined)
    print(f"  Removed duplicate pairs: {before - after}")
    
    # Remove duplicate questions (keep first)
    before = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['question'], keep='first')
    after = len(df_combined)
    print(f"  Removed duplicate questions: {before - after}")
    
    # Source breakdown
    print("\n[5] Source breakdown:")
    for src, count in df_combined['source'].value_counts().items():
        pct = count / len(df_combined) * 100
        print(f"  {src}: {count:,} ({pct:.1f}%)")
    
    # Save
    print(f"\n[6] Saving to {output_path.name}...")
    df_combined.to_csv(output_path, index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Claude Q&A:      {len(df_claude):,}")
    print(f"  Stack Exchange:  {len(df_se):,}")
    print(f"  Combined:        {len(df_combined):,}")
    print(f"\n  Output: {output_path}")
    
    print("\n" + "=" * 60)
    print("NEXT STEP")
    print("=" * 60)
    print("  python scripts/train_tokenizer_v2.py")
    
    return 0


if __name__ == "__main__":
    exit(main())
