"""
Combine Q&A Sources for v4 Training

Merges:
  - claude_qa.csv (15,000 pairs)
  - stackexchange_qa_cleaned.csv (~10K pairs)
  - CoT_Reasoning_Quantum_Physics_And_Computing.csv (3,000 pairs)

Run on HPC:
    cd ~/hpc-share/quantum-llm
    source venv/bin/activate
    python scripts/combine_qa_v4.py
"""

import pandas as pd
from pathlib import Path


def main():
    # Paths (HPC)
    data_dir = Path.home() / "hpc-share/quantum-llm/data"
    output_path = data_dir / "combined_qa_v4.csv"
    
    claude_path = data_dir / "claude_qa.csv"
    stackexchange_path = data_dir / "stackexchange_qa_cleaned.csv"
    cot_path = data_dir / "CoT_Reasoning_Quantum_Physics_And_Computing.csv"
    
    print("=" * 60)
    print("COMBINE Q&A SOURCES FOR V4")
    print("=" * 60)
    
    # Check files exist
    print("\n[1] Checking input files...")
    for path in [claude_path, stackexchange_path, cot_path]:
        if not path.exists():
            print(f"  ERROR: {path} not found")
            return 1
        print(f"  OK: {path.name}")
    
    # Load Claude Q&A
    print("\n[2] Loading Claude Q&A...")
    df_claude = pd.read_csv(claude_path)
    print(f"  Rows: {len(df_claude):,}")
    print(f"  Columns: {list(df_claude.columns)}")
    
    # Standardize columns
    df_claude = df_claude.rename(columns={
        df_claude.columns[0]: 'question',
        df_claude.columns[1]: 'answer'
    })
    df_claude['source'] = 'claude_synthetic'
    df_claude = df_claude[['question', 'answer', 'source']]
    
    # Load Stack Exchange
    print("\n[3] Loading Stack Exchange...")
    df_se = pd.read_csv(stackexchange_path)
    print(f"  Rows: {len(df_se):,}")
    print(f"  Columns: {list(df_se.columns)}")
    
    # Standardize columns
    df_se = df_se.rename(columns={
        df_se.columns[0]: 'question',
        df_se.columns[1]: 'answer'
    })
    df_se['source'] = 'stackexchange'
    df_se = df_se[['question', 'answer', 'source']]
    
    # Load CoT Reasoning
    print("\n[4] Loading CoT Reasoning...")
    # Try different delimiters
    try:
        df_cot = pd.read_csv(cot_path)
    except:
        df_cot = pd.read_csv(cot_path, sep='\t')
    
    print(f"  Rows: {len(df_cot):,}")
    print(f"  Columns: {list(df_cot.columns)}")
    
    # Standardize columns
    df_cot = df_cot.rename(columns={
        df_cot.columns[0]: 'question',
        df_cot.columns[1]: 'answer'
    })
    df_cot['source'] = 'cot_reasoning'
    df_cot = df_cot[['question', 'answer', 'source']]
    
    # Combine
    print("\n[5] Combining datasets...")
    df_combined = pd.concat([df_claude, df_se, df_cot], ignore_index=True)
    print(f"  Combined rows: {len(df_combined):,}")
    
    # Basic cleaning
    print("\n[6] Cleaning...")
    
    # Remove empty
    before = len(df_combined)
    df_combined = df_combined.dropna(subset=['question', 'answer'])
    df_combined = df_combined[df_combined['question'].astype(str).str.strip() != '']
    df_combined = df_combined[df_combined['answer'].astype(str).str.strip() != '']
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
    print("\n[7] Source breakdown:")
    for src, count in df_combined['source'].value_counts().items():
        pct = count / len(df_combined) * 100
        print(f"  {src}: {count:,} ({pct:.1f}%)")
    
    # Save
    print(f"\n[8] Saving to {output_path.name}...")
    df_combined.to_csv(output_path, index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Claude Q&A:      {len(df_claude):,}")
    print(f"  Stack Exchange:  {len(df_se):,}")
    print(f"  CoT Reasoning:   {len(df_cot):,}")
    print(f"  Combined:        {len(df_combined):,}")
    print(f"\n  Output: {output_path}")
    
    print("\n" + "=" * 60)
    print("NEXT STEP")
    print("=" * 60)
    print("  python training\\scripts\\train_tokenizer_v4.py")
    
    return 0


if __name__ == "__main__":
    exit(main())
