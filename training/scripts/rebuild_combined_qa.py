"""
Rebuild Combined Q&A Dataset

Combines:
  - chatgpt_qa_deduped.csv (35,321 rows)
  - stackexchange_qa_cleaned.csv (10,662 rows)

Then filters out examples >1024 tokens (truncated garbage).

Run on HPC:
    cd ~/hpc-share/quantum-llm
    source venv/bin/activate
    python scripts/rebuild_combined_qa.py
"""

import pandas as pd
from pathlib import Path


def main():
    # Paths (HPC)
    data_dir = Path.home() / "hpc-share/quantum-llm/data"
    chatgpt_path = data_dir / "chatgpt_qa_deduped.csv"
    stackexchange_path = data_dir / "stackexchange_qa_cleaned.csv"
    output_path = data_dir / "combined_qa_final_v3.csv"
    tokenizer_path = Path.home() / "hpc-share/quantum-llm/tokenizer.json"
    
    MAX_TOKENS = 1024
    
    print("=" * 60)
    print("REBUILD COMBINED Q&A DATASET")
    print("=" * 60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    
    # Load ChatGPT data
    print(f"\nLoading {chatgpt_path.name}...")
    df_chatgpt = pd.read_csv(chatgpt_path)
    print(f"  Rows: {len(df_chatgpt):,}")
    
    # Standardize columns
    df_chatgpt = df_chatgpt.rename(columns={
        df_chatgpt.columns[0]: 'question',
        df_chatgpt.columns[1]: 'answer'
    })
    df_chatgpt['source'] = 'chatgpt_synthetic'
    
    # Load Stack Exchange data
    print(f"\nLoading {stackexchange_path.name}...")
    df_se = pd.read_csv(stackexchange_path)
    print(f"  Rows: {len(df_se):,}")
    
    # Standardize columns
    df_se = df_se.rename(columns={
        df_se.columns[0]: 'question',
        df_se.columns[1]: 'answer'
    })
    df_se['source'] = 'stackexchange'
    
    # Combine
    print("\nCombining datasets...")
    df_combined = pd.concat([df_chatgpt, df_se], ignore_index=True)
    print(f"  Combined rows: {len(df_combined):,}")
    
    # Calculate token lengths
    print(f"\nCalculating token lengths...")
    
    def get_token_count(row):
        formatted = f"Q: {row['question']}\nA: {row['answer']}"
        return len(tokenizer.encode(formatted).ids) + 1  # +1 for EOS
    
    df_combined['tokens'] = df_combined.apply(get_token_count, axis=1)
    
    # Stats before filtering
    print(f"\n  Token length stats:")
    print(f"    Min: {df_combined['tokens'].min()}")
    print(f"    Max: {df_combined['tokens'].max()}")
    print(f"    Mean: {df_combined['tokens'].mean():.1f}")
    print(f"    Median: {df_combined['tokens'].median():.0f}")
    
    over_limit = (df_combined['tokens'] > MAX_TOKENS).sum()
    print(f"\n  Examples > {MAX_TOKENS} tokens: {over_limit:,} ({over_limit/len(df_combined)*100:.1f}%)")
    
    # Show what we're removing
    print(f"\nExamples being removed (>{MAX_TOKENS} tokens):")
    removed = df_combined[df_combined['tokens'] > MAX_TOKENS]
    by_source = removed['source'].value_counts()
    for src, count in by_source.items():
        print(f"    {src}: {count:,}")
    
    # Filter
    print(f"\nFiltering examples > {MAX_TOKENS} tokens...")
    df_filtered = df_combined[df_combined['tokens'] <= MAX_TOKENS].copy()
    df_filtered = df_filtered.drop(columns=['tokens'])
    
    print(f"  Removed: {len(df_combined) - len(df_filtered):,} rows")
    print(f"  Remaining: {len(df_filtered):,} rows")
    
    # Final source breakdown
    print("\nFinal source breakdown:")
    for src, count in df_filtered['source'].value_counts().items():
        print(f"    {src}: {count:,} ({count/len(df_filtered)*100:.1f}%)")
    
    # Save
    print(f"\nSaving to {output_path}...")
    df_filtered.to_csv(output_path, index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  ChatGPT (deduped):    {len(df_chatgpt):,}")
    print(f"  Stack Exchange:       {len(df_se):,}")
    print(f"  Combined:             {len(df_combined):,}")
    print(f"  After filtering:      {len(df_filtered):,}")
    print(f"\nOutput: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
