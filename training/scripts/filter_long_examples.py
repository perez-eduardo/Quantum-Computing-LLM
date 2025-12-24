"""
Filter Long Examples

Removes Q&A pairs that exceed 1024 tokens (would be severely truncated).
Uses the newly trained tokenizer.

Run on HPC:
    cd ~/hpc-share/quantum-llm
    source venv/bin/activate
    python scripts/filter_long_examples.py
"""

import pandas as pd
from pathlib import Path
from tokenizers import Tokenizer


MAX_TOKENS = 1024  # Examples above this get removed


def main():
    data_dir = Path.home() / "hpc-share/quantum-llm/data"
    tokenizer_path = Path.home() / "hpc-share/quantum-llm/tokenizer.json"
    
    input_path = data_dir / "combined_qa.csv"
    output_path = data_dir / "combined_qa_filtered.csv"
    
    print("=" * 60)
    print("FILTER LONG EXAMPLES")
    print("=" * 60)
    print(f"\nRemoving examples > {MAX_TOKENS} tokens")
    print("(These would be severely truncated during training)")
    
    # Load tokenizer
    print("\n[1] Loading tokenizer...")
    if not tokenizer_path.exists():
        print(f"  ERROR: {tokenizer_path} not found")
        print("  Run train_tokenizer_v2.py first!")
        return 1
    
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"  Vocab size: {tokenizer.get_vocab_size():,}")
    
    # Load data
    print(f"\n[2] Loading {input_path.name}...")
    if not input_path.exists():
        print(f"  ERROR: {input_path} not found")
        return 1
    
    df = pd.read_csv(input_path)
    print(f"  Rows: {len(df):,}")
    
    # Calculate token lengths
    print("\n[3] Calculating token lengths...")
    
    def get_token_count(row):
        formatted = f"Q: {row['question']}\nA: {row['answer']}"
        return len(tokenizer.encode(formatted).ids) + 1  # +1 for EOS
    
    df['tokens'] = df.apply(get_token_count, axis=1)
    
    # Stats before filtering
    print("\n  Token length stats (before filtering):")
    print(f"    Min:    {df['tokens'].min()}")
    print(f"    Max:    {df['tokens'].max()}")
    print(f"    Mean:   {df['tokens'].mean():.1f}")
    print(f"    Median: {df['tokens'].median():.0f}")
    print(f"    P95:    {df['tokens'].quantile(0.95):.0f}")
    print(f"    P99:    {df['tokens'].quantile(0.99):.0f}")
    
    # Count over limit
    over_limit = df[df['tokens'] > MAX_TOKENS]
    print(f"\n  Examples > {MAX_TOKENS} tokens: {len(over_limit):,} ({len(over_limit)/len(df)*100:.1f}%)")
    
    # Breakdown by source
    print("\n  By source:")
    for src in df['source'].unique():
        src_df = df[df['source'] == src]
        src_over = src_df[src_df['tokens'] > MAX_TOKENS]
        print(f"    {src}: {len(src_over):,} / {len(src_df):,} over limit")
    
    # Show how much over
    if len(over_limit) > 0:
        print("\n  How much over the limit:")
        over_amounts = over_limit['tokens'] - MAX_TOKENS
        print(f"    Min over:  {over_amounts.min()}")
        print(f"    Max over:  {over_amounts.max()}")
        print(f"    Mean over: {over_amounts.mean():.0f}")
    
    # Filter
    print(f"\n[4] Filtering examples > {MAX_TOKENS} tokens...")
    df_filtered = df[df['tokens'] <= MAX_TOKENS].copy()
    df_filtered = df_filtered.drop(columns=['tokens'])
    
    print(f"  Removed: {len(df) - len(df_filtered):,}")
    print(f"  Remaining: {len(df_filtered):,}")
    
    # Final source breakdown
    print("\n[5] Final source breakdown:")
    for src, count in df_filtered['source'].value_counts().items():
        pct = count / len(df_filtered) * 100
        print(f"  {src}: {count:,} ({pct:.1f}%)")
    
    # Save
    print(f"\n[6] Saving to {output_path.name}...")
    df_filtered.to_csv(output_path, index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Input:    {len(df):,} rows")
    print(f"  Removed:  {len(df) - len(df_filtered):,} rows (>{MAX_TOKENS} tokens)")
    print(f"  Output:   {len(df_filtered):,} rows")
    print(f"\n  File: {output_path}")
    
    print("\n" + "=" * 60)
    print("NEXT STEP")
    print("=" * 60)
    print("  python scripts/verify_dataset_v2.py")
    
    return 0


if __name__ == "__main__":
    exit(main())
