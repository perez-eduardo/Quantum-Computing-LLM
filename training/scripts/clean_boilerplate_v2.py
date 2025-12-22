"""
Clean Boilerplate V2 - Strip repeated sentences from answers

Strategy:
1. Find all sentences appearing 100+ times across answers
2. Remove those sentences from each answer
3. Drop answers with <20 words remaining

Run on HPC:
    cd ~/hpc-share/quantum-llm
    source venv/bin/activate
    python scripts/clean_boilerplate_v2.py
"""

import pandas as pd
import re
from collections import Counter
from pathlib import Path


BOILERPLATE_THRESHOLD = 100  # Sentences appearing this many times = boilerplate
MIN_WORDS_AFTER_STRIP = 20   # Minimum words to keep an answer


def extract_sentences(text):
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', str(text))
    return [s.strip() for s in sentences if s.strip()]


def count_words(text):
    """Count words in text."""
    return len(str(text).split())


def main():
    data_path = Path.home() / "hpc-share/quantum-llm/data/combined_qa_final_v3.csv"
    output_path = Path.home() / "hpc-share/quantum-llm/data/combined_qa_final_v4.csv"
    
    print("=" * 70)
    print("CLEAN BOILERPLATE V2 - STRIP REPEATED SENTENCES")
    print("=" * 70)
    print(f"\n  Boilerplate threshold: {BOILERPLATE_THRESHOLD}+ occurrences")
    print(f"  Min words after strip: {MIN_WORDS_AFTER_STRIP}")
    
    # Load data
    print(f"\nLoading {data_path.name}...")
    df = pd.read_csv(data_path)
    print(f"  Rows: {len(df):,}")
    
    q_col = df.columns[0]
    a_col = df.columns[1]
    
    # Step 1: Find boilerplate sentences
    print("\n" + "-" * 70)
    print("STEP 1: Identify boilerplate sentences")
    print("-" * 70)
    
    all_sentences = []
    for answer in df[a_col]:
        all_sentences.extend(extract_sentences(answer))
    
    print(f"  Total sentences extracted: {len(all_sentences):,}")
    
    sentence_counts = Counter(all_sentences)
    boilerplate_sentences = {s for s, c in sentence_counts.items() if c >= BOILERPLATE_THRESHOLD}
    
    print(f"  Boilerplate sentences ({BOILERPLATE_THRESHOLD}+ occurrences): {len(boilerplate_sentences)}")
    
    # Show top 10
    top_boilerplate = sorted([(s, sentence_counts[s]) for s in boilerplate_sentences], key=lambda x: -x[1])[:10]
    print("\n  Top 10 boilerplate sentences:")
    for i, (sent, count) in enumerate(top_boilerplate, 1):
        display = sent[:60] + "..." if len(sent) > 60 else sent
        print(f"    {i:2}. [{count:,}x] {display}")
    
    # Step 2: Strip boilerplate from answers
    print("\n" + "-" * 70)
    print("STEP 2: Strip boilerplate sentences from answers")
    print("-" * 70)
    
    def strip_boilerplate(answer):
        sentences = extract_sentences(answer)
        cleaned = [s for s in sentences if s not in boilerplate_sentences]
        return ' '.join(cleaned)
    
    df['answer_cleaned'] = df[a_col].apply(strip_boilerplate)
    df['words_before'] = df[a_col].apply(count_words)
    df['words_after'] = df['answer_cleaned'].apply(count_words)
    df['words_removed'] = df['words_before'] - df['words_after']
    
    # Stats
    affected = (df['words_removed'] > 0).sum()
    total_words_removed = df['words_removed'].sum()
    
    print(f"  Answers affected: {affected:,} ({affected/len(df)*100:.1f}%)")
    print(f"  Total words removed: {total_words_removed:,}")
    
    # Step 3: Filter short answers
    print("\n" + "-" * 70)
    print("STEP 3: Filter answers with <{} words".format(MIN_WORDS_AFTER_STRIP))
    print("-" * 70)
    
    before_filter = len(df)
    too_short = df[df['words_after'] < MIN_WORDS_AFTER_STRIP]
    
    print(f"  Answers becoming too short: {len(too_short):,}")
    
    # Show examples of what's being removed
    print("\n  Examples of removed (too short after cleaning):")
    samples = too_short.head(5)
    for idx, row in samples.iterrows():
        print(f"\n    BEFORE ({row['words_before']} words): {row[a_col][:100]}...")
        print(f"    AFTER ({row['words_after']} words): {row['answer_cleaned'][:100] if row['answer_cleaned'] else '[EMPTY]'}")
    
    # Filter
    df_clean = df[df['words_after'] >= MIN_WORDS_AFTER_STRIP].copy()
    
    print(f"\n  Before filter: {before_filter:,}")
    print(f"  After filter: {len(df_clean):,}")
    print(f"  Removed: {before_filter - len(df_clean):,}")
    
    # Step 4: Prepare output
    print("\n" + "-" * 70)
    print("STEP 4: Prepare output")
    print("-" * 70)
    
    # Replace answer column with cleaned version
    df_clean[a_col] = df_clean['answer_cleaned']
    
    # Keep only original columns
    output_cols = [c for c in df.columns if c not in ['answer_cleaned', 'words_before', 'words_after', 'words_removed']]
    df_output = df_clean[output_cols]
    
    # Source breakdown
    if 'source' in df_output.columns:
        print("\n  Source breakdown:")
        for source, count in df_output['source'].value_counts().items():
            pct = count / len(df_output) * 100
            print(f"    {source}: {count:,} ({pct:.1f}%)")
    
    # Save
    print(f"\nSaving to {output_path.name}...")
    df_output.to_csv(output_path, index=False)
    
    # Verification
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    # Check a few random samples
    print("\nRandom cleaned samples:")
    samples = df_output.sample(n=5, random_state=42)
    for i, (idx, row) in enumerate(samples.iterrows(), 1):
        q = row[q_col][:80] + "..." if len(str(row[q_col])) > 80 else row[q_col]
        a = row[a_col][:120] + "..." if len(str(row[a_col])) > 120 else row[a_col]
        print(f"\n  [{i}] Q: {q}")
        print(f"      A: {a}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Input rows:           {before_filter:,}")
    print(f"  Boilerplate patterns: {len(boilerplate_sentences)}")
    print(f"  Answers affected:     {affected:,}")
    print(f"  Removed (too short):  {before_filter - len(df_output):,}")
    print(f"  Output rows:          {len(df_output):,}")
    print(f"\n  Output: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
