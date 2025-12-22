"""
Find Additional Boilerplate Patterns

Scans for repetitive phrases that appear across many answers.
Focuses on sentence endings and appended clauses.

Run on HPC:
    cd ~/hpc-share/quantum-llm
    source venv/bin/activate
    python scripts/find_boilerplate_v2.py
"""

import pandas as pd
import re
from collections import Counter
from pathlib import Path


def extract_sentences(text):
    """Split text into sentences."""
    # Simple sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+', str(text))
    return [s.strip() for s in sentences if s.strip()]


def main():
    # Path (HPC)
    data_path = Path.home() / "hpc-share/quantum-llm/data/combined_qa_final_v3.csv"
    
    print("=" * 70)
    print("FIND ADDITIONAL BOILERPLATE PATTERNS")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading {data_path.name}...")
    df = pd.read_csv(data_path)
    print(f"  Rows: {len(df):,}")
    
    # Detect columns
    q_col = df.columns[0]
    a_col = df.columns[1]
    print(f"  Answer column: '{a_col}'")
    
    # Extract all sentences from answers
    print("\nExtracting sentences from answers...")
    all_sentences = []
    for answer in df[a_col]:
        sentences = extract_sentences(answer)
        all_sentences.extend(sentences)
    
    print(f"  Total sentences: {len(all_sentences):,}")
    
    # Count sentence frequency
    print("\nCounting sentence frequencies...")
    sentence_counts = Counter(all_sentences)
    
    # Find sentences that appear 10+ times
    repeated = {s: c for s, c in sentence_counts.items() if c >= 10}
    print(f"  Sentences appearing 10+ times: {len(repeated):,}")
    
    # Sort by frequency
    sorted_repeated = sorted(repeated.items(), key=lambda x: -x[1])
    
    # Show top 50
    print("\n" + "=" * 70)
    print("TOP 50 REPEATED SENTENCES (potential boilerplate)")
    print("=" * 70)
    
    for i, (sentence, count) in enumerate(sorted_repeated[:50], 1):
        display = sentence[:80] + "..." if len(sentence) > 80 else sentence
        print(f"\n{i:2}. [{count:,}x] {display}")
    
    # Find repeated ending patterns (last sentence of answers)
    print("\n" + "=" * 70)
    print("REPEATED ANSWER ENDINGS (last sentence)")
    print("=" * 70)
    
    last_sentences = []
    for answer in df[a_col]:
        sentences = extract_sentences(answer)
        if sentences:
            last_sentences.append(sentences[-1])
    
    ending_counts = Counter(last_sentences)
    repeated_endings = {s: c for s, c in ending_counts.items() if c >= 5}
    sorted_endings = sorted(repeated_endings.items(), key=lambda x: -x[1])
    
    print(f"\nEndings appearing 5+ times: {len(sorted_endings):,}")
    
    for i, (sentence, count) in enumerate(sorted_endings[:30], 1):
        display = sentence[:80] + "..." if len(sentence) > 80 else sentence
        print(f"\n{i:2}. [{count:,}x] {display}")
    
    # Find common phrase patterns (partial matches)
    print("\n" + "=" * 70)
    print("COMMON PHRASE PATTERNS")
    print("=" * 70)
    
    # Known suspicious patterns to check
    patterns_to_check = [
        "If results differ",
        "Real performance is",
        "check transpilation",
        "calibration quality",
        "system-level integration",
        "coherence times",
        "qubit mapping",
        "readout calibration",
        "On real hardware",
        "In practice",
        "For real devices",
        "When running on",
        "The actual behavior",
        "Actual results may",
        "Performance depends on",
    ]
    
    print("\nChecking for known suspicious patterns:")
    for pattern in patterns_to_check:
        count = df[a_col].str.contains(pattern, case=False, na=False).sum()
        if count > 0:
            pct = count / len(df) * 100
            print(f"  [{count:,}] ({pct:.1f}%) '{pattern}'")
    
    # Find all phrases starting with "If " that appear often
    print("\n" + "=" * 70)
    print("SENTENCES STARTING WITH 'If ' (frequent)")
    print("=" * 70)
    
    if_sentences = [s for s in all_sentences if s.startswith("If ")]
    if_counts = Counter(if_sentences)
    repeated_if = sorted([(s, c) for s, c in if_counts.items() if c >= 5], key=lambda x: -x[1])
    
    for i, (sentence, count) in enumerate(repeated_if[:20], 1):
        display = sentence[:80] + "..." if len(sentence) > 80 else sentence
        print(f"\n{i:2}. [{count:,}x] {display}")
    
    # Save full report
    output_path = Path.home() / "hpc-share/quantum-llm/data/boilerplate_v2_report.txt"
    print(f"\n" + "=" * 70)
    print(f"Saving full report to: {output_path}")
    print("=" * 70)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("BOILERPLATE V2 ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("ALL SENTENCES APPEARING 10+ TIMES:\n")
        f.write("-" * 70 + "\n")
        for sentence, count in sorted_repeated:
            f.write(f"[{count}x] {sentence}\n\n")
        
        f.write("\n\nALL ENDINGS APPEARING 5+ TIMES:\n")
        f.write("-" * 70 + "\n")
        for sentence, count in sorted_endings:
            f.write(f"[{count}x] {sentence}\n\n")
    
    print("Done. Review boilerplate_v2_report.txt for full list.")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_boilerplate_candidates = sum(c for s, c in sorted_repeated[:20])
    print(f"\n  Top 20 repeated sentences account for: {total_boilerplate_candidates:,} occurrences")
    print(f"  This may indicate additional boilerplate to clean.")
    
    return 0


if __name__ == "__main__":
    exit(main())
