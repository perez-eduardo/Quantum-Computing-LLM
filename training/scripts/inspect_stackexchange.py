"""
Inspect Stack Exchange Q&A Data

Detailed inspection of stackexchange_qa.csv for quality issues.

Usage:
    python inspect_stackexchange.py --input data/raw/stackexchange_qa.csv
"""

import argparse
import re
from pathlib import Path
from collections import Counter
import pandas as pd


def find_html_entities(text: str) -> list[str]:
    """Find HTML entities in text."""
    return re.findall(r'&[a-zA-Z]+;|&#\d+;', str(text))


def find_html_tags(text: str) -> list[str]:
    """Find actual HTML tags (not LaTeX)."""
    # Real HTML tags like <p>, <code>, <a href>, etc.
    tags = re.findall(r'</?(?:p|div|span|code|pre|a|b|i|strong|em|br|hr|ul|ol|li|table|tr|td|th|img|h[1-6])[^>]*>', str(text), re.IGNORECASE)
    return tags


def find_code_blocks(text: str) -> list[str]:
    """Find code blocks."""
    return re.findall(r'```[\s\S]*?```|`[^`]+`', str(text))


def main():
    parser = argparse.ArgumentParser(description="Inspect Stack Exchange Q&A")
    parser.add_argument("--input", required=True, help="Path to stackexchange_qa.csv")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: {args.input} not found")
        return 1
    
    print("=" * 60)
    print("STACK EXCHANGE DATA INSPECTION")
    print("=" * 60)
    
    df = pd.read_csv(args.input)
    
    # Basic info
    print(f"\nRows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    # Detect column names
    cols = df.columns.tolist()
    q_col = cols[0]
    a_col = cols[1]
    print(f"Question column: '{q_col}'")
    print(f"Answer column: '{a_col}'")
    
    # Null/empty
    print("\n--- NULL/EMPTY ---")
    print(f"Null questions: {df[q_col].isna().sum()}")
    print(f"Null answers: {df[a_col].isna().sum()}")
    print(f"Empty questions: {(df[q_col].astype(str).str.strip() == '').sum()}")
    print(f"Empty answers: {(df[a_col].astype(str).str.strip() == '').sum()}")
    
    # Length stats
    print("\n--- LENGTH STATS (characters) ---")
    df['q_len'] = df[q_col].astype(str).str.len()
    df['a_len'] = df[a_col].astype(str).str.len()
    
    print(f"Question: min={df['q_len'].min()}, median={df['q_len'].median():.0f}, max={df['q_len'].max()}")
    print(f"Answer: min={df['a_len'].min()}, median={df['a_len'].median():.0f}, max={df['a_len'].max()}")
    
    print(f"\nVery short questions (<20 chars): {(df['q_len'] < 20).sum()}")
    print(f"Very short answers (<50 chars): {(df['a_len'] < 50).sum()}")
    print(f"Very long questions (>2000 chars): {(df['q_len'] > 2000).sum()}")
    print(f"Very long answers (>5000 chars): {(df['a_len'] > 5000).sum()}")
    
    # HTML entities
    print("\n--- HTML ENTITIES ---")
    all_entities = []
    rows_with_entities = 0
    for _, row in df.iterrows():
        text = str(row[q_col]) + " " + str(row[a_col])
        entities = find_html_entities(text)
        if entities:
            rows_with_entities += 1
            all_entities.extend(entities)
    
    print(f"Rows with HTML entities: {rows_with_entities} ({rows_with_entities/len(df)*100:.1f}%)")
    if all_entities:
        entity_counts = Counter(all_entities).most_common(15)
        print("Most common entities:")
        for ent, count in entity_counts:
            print(f"  {ent}: {count}")
    
    # HTML tags
    print("\n--- HTML TAGS ---")
    all_tags = []
    rows_with_tags = 0
    for _, row in df.iterrows():
        text = str(row[q_col]) + " " + str(row[a_col])
        tags = find_html_tags(text)
        if tags:
            rows_with_tags += 1
            all_tags.extend(tags)
    
    print(f"Rows with HTML tags: {rows_with_tags} ({rows_with_tags/len(df)*100:.1f}%)")
    if all_tags:
        tag_counts = Counter(all_tags).most_common(10)
        print("Most common tags:")
        for tag, count in tag_counts:
            print(f"  {tag}: {count}")
    
    # Code blocks
    print("\n--- CODE BLOCKS ---")
    rows_with_code = 0
    for _, row in df.iterrows():
        text = str(row[q_col]) + " " + str(row[a_col])
        if find_code_blocks(text):
            rows_with_code += 1
    print(f"Rows with code blocks: {rows_with_code} ({rows_with_code/len(df)*100:.1f}%)")
    
    # Duplicates
    print("\n--- DUPLICATES ---")
    print(f"Duplicate questions: {df[q_col].duplicated().sum()}")
    print(f"Duplicate answers: {df[a_col].duplicated().sum()}")
    
    # Sample rows with issues
    print("\n--- SAMPLE ROWS WITH HTML ENTITIES (up to 5) ---")
    shown = 0
    for _, row in df.iterrows():
        text = str(row[q_col]) + " " + str(row[a_col])
        entities = find_html_entities(text)
        if entities:
            print(f"\nQ: {str(row[q_col])[:100]}...")
            print(f"A: {str(row[a_col])[:100]}...")
            print(f"Entities found: {entities[:10]}")
            shown += 1
            if shown >= 5:
                break
    
    # Sample rows with HTML tags
    print("\n--- SAMPLE ROWS WITH HTML TAGS (up to 5) ---")
    shown = 0
    for _, row in df.iterrows():
        text = str(row[q_col]) + " " + str(row[a_col])
        tags = find_html_tags(text)
        if tags:
            print(f"\nQ: {str(row[q_col])[:100]}...")
            print(f"A: {str(row[a_col])[:150]}...")
            print(f"Tags found: {tags[:10]}")
            shown += 1
            if shown >= 5:
                break
    
    if shown == 0:
        print("None found!")
    
    # Random clean samples
    print("\n--- RANDOM SAMPLES (5) ---")
    samples = df.sample(5, random_state=42)
    for i, (_, row) in enumerate(samples.iterrows(), 1):
        print(f"\n[Sample {i}]")
        print(f"Q: {str(row[q_col])[:150]}")
        print(f"A: {str(row[a_col])[:200]}...")
    
    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
