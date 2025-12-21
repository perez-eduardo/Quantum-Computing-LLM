"""
Combine Q&A Sources

Merges ChatGPT, Stack Exchange, and QuantumLLMInstruct data into combined_qa.csv.

Usage:
    python combine_qa.py --chatgpt data/raw/chatgpt_qa_cleaned.csv --stackexchange data/raw/stackexchange_qa.csv --quantumllm data/raw/quantumllm_instruct.csv --output data/raw/combined_qa.csv

Requirements:
    pip install pandas
"""

import argparse
import pandas as pd
from pathlib import Path


def load_and_tag(path: str, source_name: str, q_col: str = None, a_col: str = None) -> pd.DataFrame:
    """Load CSV and standardize columns."""
    if not Path(path).exists():
        print(f"  Warning: {path} not found, skipping")
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    print(f"  {source_name}: {len(df):,} rows")
    
    # Detect column names if not specified
    cols = df.columns.tolist()
    
    if q_col is None:
        q_col = next((c for c in cols if 'question' in c.lower() or c.lower() == 'q'), cols[0])
    if a_col is None:
        a_col = next((c for c in cols if 'answer' in c.lower() or c.lower() == 'a'), cols[1])
    
    # Standardize
    result = pd.DataFrame({
        'question': df[q_col].astype(str).str.strip(),
        'answer': df[a_col].astype(str).str.strip(),
        'source': source_name
    })
    
    # Remove empty rows
    result = result[(result['question'].str.len() > 0) & (result['answer'].str.len() > 0)]
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Combine Q&A sources")
    parser.add_argument("--chatgpt", required=True, help="Path to cleaned ChatGPT CSV")
    parser.add_argument("--stackexchange", required=True, help="Path to Stack Exchange CSV")
    parser.add_argument("--quantumllm", help="Path to QuantumLLMInstruct CSV (optional)")
    parser.add_argument("--output", required=True, help="Output path for combined CSV")
    
    args = parser.parse_args()
    
    print("Loading sources...")
    
    dfs = []
    
    # ChatGPT
    df_chatgpt = load_and_tag(args.chatgpt, "chatgpt_synthetic", q_col="Question", a_col="Answer")
    if len(df_chatgpt) > 0:
        dfs.append(df_chatgpt)
    
    # Stack Exchange
    df_se = load_and_tag(args.stackexchange, "stackexchange")
    if len(df_se) > 0:
        dfs.append(df_se)
    
    # QuantumLLMInstruct (optional)
    if args.quantumllm and Path(args.quantumllm).exists():
        df_qllm = load_and_tag(args.quantumllm, "quantumllm_instruct")
        if len(df_qllm) > 0:
            dfs.append(df_qllm)
    
    if not dfs:
        print("Error: No data loaded!")
        return 1
    
    # Combine
    print("\nCombining...")
    combined = pd.concat(dfs, ignore_index=True)
    
    # Remove exact duplicates
    before = len(combined)
    combined = combined.drop_duplicates(subset=['question', 'answer'])
    after = len(combined)
    print(f"Removed {before - after:,} exact duplicate Q&A pairs")
    
    # Save
    combined.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")
    print(f"Total rows: {len(combined):,}")
    
    # Summary
    print("\n--- SOURCE BREAKDOWN ---")
    print(combined['source'].value_counts())
    
    # Estimate tokens
    total_chars = combined['question'].str.len().sum() + combined['answer'].str.len().sum()
    est_tokens = total_chars / 4  # Rough estimate: 4 chars per token
    print(f"\nEstimated tokens: ~{est_tokens/1e6:.1f}M")
    
    return 0


if __name__ == "__main__":
    exit(main())
