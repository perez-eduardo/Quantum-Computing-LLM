"""
Inspect ChatGPT Q&A data for templated/repetitive questions.
Identifies patterns like "Why does X with Y qubits?" repeated with different numbers.

Run from: E:\Personal_projects\Quantum-Computing-LLM\
Usage: python inspect_templates.py
"""

import pandas as pd
import re
from collections import Counter, defaultdict
from pathlib import Path


def normalize_question(q):
    """Replace numbers with placeholder to find templates."""
    # Replace numbers (including decimals) with <NUM>
    normalized = re.sub(r'\b\d+\.?\d*\b', '<NUM>', str(q))
    return normalized.strip()


def main():
    # Path to ChatGPT data (HPC)
    data_path = Path.home() / "hpc-share/quantum-llm/data/chatgpt_qa_cleaned_v2.csv"
    
    if not data_path.exists():
        print(f"ERROR: File not found: {data_path}")
        return 1
    
    print("=" * 70)
    print("TEMPLATE INSPECTION: chatgpt_qa_cleaned_v2.csv")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"\nTotal rows: {len(df):,}")
    
    # Detect column names dynamically
    q_col = df.columns[0]
    a_col = df.columns[1]
    print(f"Question column: '{q_col}'")
    print(f"Answer column: '{a_col}'")
    
    # Normalize questions (replace numbers with placeholder)
    print("\nNormalizing questions (replacing numbers with <NUM>)...")
    df['normalized'] = df[q_col].apply(normalize_question)
    
    # Count occurrences of each normalized pattern
    pattern_counts = Counter(df['normalized'])
    
    # Find templates (patterns that appear more than once)
    templates = {k: v for k, v in pattern_counts.items() if v > 1}
    
    print(f"\nUnique questions: {len(pattern_counts):,}")
    print(f"Templated patterns (>1 occurrence): {len(templates):,}")
    
    # Calculate total templated rows
    templated_rows = sum(v for v in templates.values())
    unique_rows = len(df) - templated_rows + len(templates)  # keep 1 of each template
    
    print(f"\nRows from templates: {templated_rows:,} ({templated_rows/len(df)*100:.1f}%)")
    print(f"If deduplicated (keep 1 each): {unique_rows:,} rows remaining")
    
    # Sort by frequency
    sorted_templates = sorted(templates.items(), key=lambda x: -x[1])
    
    # Show top 30 templates
    print("\n" + "=" * 70)
    print("TOP 30 TEMPLATES (by frequency)")
    print("=" * 70)
    
    for i, (pattern, count) in enumerate(sorted_templates[:30], 1):
        # Truncate long patterns
        display = pattern[:80] + "..." if len(pattern) > 80 else pattern
        print(f"\n{i:2}. [{count:,}x] {display}")
        
        # Show 2 actual examples
        examples = df[df['normalized'] == pattern][q_col].head(2).tolist()
        for ex in examples:
            ex_display = ex[:70] + "..." if len(ex) > 70 else ex
            print(f"      Example: {ex_display}")
    
    # Frequency distribution
    print("\n" + "=" * 70)
    print("TEMPLATE FREQUENCY DISTRIBUTION")
    print("=" * 70)
    
    freq_buckets = defaultdict(int)
    for count in templates.values():
        if count >= 100:
            freq_buckets["100+"] += count
        elif count >= 50:
            freq_buckets["50-99"] += count
        elif count >= 20:
            freq_buckets["20-49"] += count
        elif count >= 10:
            freq_buckets["10-19"] += count
        elif count >= 5:
            freq_buckets["5-9"] += count
        else:
            freq_buckets["2-4"] += count
    
    print(f"\n{'Frequency':<15} {'Rows':<12} {'Percent':<10}")
    print("-" * 40)
    for bucket in ["100+", "50-99", "20-49", "10-19", "5-9", "2-4"]:
        if bucket in freq_buckets:
            rows = freq_buckets[bucket]
            pct = rows / len(df) * 100
            print(f"{bucket:<15} {rows:<12,} {pct:.1f}%")
    
    # Non-templated rows
    non_templated = len(df) - templated_rows
    print(f"{'Unique (1x)':<15} {non_templated:<12,} {non_templated/len(df)*100:.1f}%")
    
    # Save full template list for review
    output_path = Path.home() / "hpc-share/quantum-llm/data/template_patterns.txt"
    print(f"\n" + "=" * 70)
    print(f"Saving all {len(sorted_templates):,} patterns to: {output_path}")
    print("=" * 70)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("TEMPLATE PATTERNS IN CHATGPT DATA\n")
        f.write(f"Total: {len(sorted_templates):,} patterns\n")
        f.write("=" * 70 + "\n\n")
        
        for i, (pattern, count) in enumerate(sorted_templates, 1):
            f.write(f"{i}. [{count}x] {pattern}\n")
            
            # Include 3 examples for each
            examples = df[df['normalized'] == pattern][q_col].head(3).tolist()
            for ex in examples:
                f.write(f"   - {ex}\n")
            f.write("\n")
    
    print("Done. Review template_patterns.txt for full list.")
    
    # Summary recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    
    high_freq = sum(1 for c in templates.values() if c >= 10)
    high_freq_rows = sum(c for c in templates.values() if c >= 10)
    
    print(f"""
  Templates with 10+ copies: {high_freq:,} patterns ({high_freq_rows:,} rows)
  
  Suggested approach:
    1. Review template_patterns.txt
    2. Decide: keep 1, 2, or 0 copies of each template
    3. Run deduplication script
    
  If keeping 1 copy of each template:
    - Remove: {templated_rows - len(templates):,} rows
    - Remaining: {unique_rows:,} rows
""")
    
    return 0


if __name__ == "__main__":
    exit(main())
