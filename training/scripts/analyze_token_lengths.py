"""
Analyze Token Lengths in Training Data

Checks if Q&A pairs are getting truncated at 512 tokens.
Run locally where you have the data files.

Usage:
    python analyze_token_lengths.py
"""

import csv
from pathlib import Path
from collections import Counter

# Paths - adjust if needed
QA_PATH = Path("data/raw/combined_qa_final.csv")
TOKENIZER_PATH = Path("training/tokenizer/tokenizer.json")
MAX_SEQ_LEN = 512


def main():
    print("=" * 60)
    print("TOKEN LENGTH ANALYSIS")
    print("=" * 60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    except ImportError:
        print("Error: tokenizers library not installed")
        print("Run: pip install tokenizers")
        return 1
    
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    
    # Load Q&A data
    print(f"\nLoading Q&A data from {QA_PATH}...")
    qa_pairs = []
    with open(QA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get('question', '').strip()
            a = row.get('answer', '').strip()
            if q and a:
                qa_pairs.append((q, a))
    
    print(f"  Loaded {len(qa_pairs):,} Q&A pairs")
    
    # Analyze token lengths
    print("\nAnalyzing token lengths...")
    
    q_lengths = []
    a_lengths = []
    combined_lengths = []
    truncated_count = 0
    
    for q, a in qa_pairs:
        # Format as training sees it: "Q: {question}\nA: {answer}<eos>"
        formatted = f"Q: {q}\nA: {a}"
        
        q_tokens = len(tokenizer.encode(f"Q: {q}").ids)
        a_tokens = len(tokenizer.encode(f"A: {a}").ids)
        combined_tokens = len(tokenizer.encode(formatted).ids) + 1  # +1 for EOS
        
        q_lengths.append(q_tokens)
        a_lengths.append(a_tokens)
        combined_lengths.append(combined_tokens)
        
        if combined_tokens > MAX_SEQ_LEN:
            truncated_count += 1
    
    # Statistics
    def stats(lengths, name):
        lengths_sorted = sorted(lengths)
        n = len(lengths)
        print(f"\n{name}:")
        print(f"  Min:    {min(lengths)}")
        print(f"  Max:    {max(lengths)}")
        print(f"  Mean:   {sum(lengths)/n:.1f}")
        print(f"  Median: {lengths_sorted[n//2]}")
        print(f"  P90:    {lengths_sorted[int(n*0.9)]}")
        print(f"  P95:    {lengths_sorted[int(n*0.95)]}")
        print(f"  P99:    {lengths_sorted[int(n*0.99)]}")
    
    stats(q_lengths, "QUESTION TOKENS")
    stats(a_lengths, "ANSWER TOKENS")
    stats(combined_lengths, "COMBINED TOKENS (Q + A + formatting)")
    
    # Truncation analysis
    print("\n" + "=" * 60)
    print("TRUNCATION ANALYSIS")
    print("=" * 60)
    print(f"\nMax sequence length: {MAX_SEQ_LEN}")
    print(f"Pairs exceeding limit: {truncated_count:,} ({truncated_count/len(qa_pairs)*100:.1f}%)")
    
    # Distribution of truncated amounts
    over_limit = [l - MAX_SEQ_LEN for l in combined_lengths if l > MAX_SEQ_LEN]
    if over_limit:
        print(f"\nHow much over the limit:")
        print(f"  Min over:  {min(over_limit)} tokens")
        print(f"  Max over:  {max(over_limit)} tokens")
        print(f"  Mean over: {sum(over_limit)/len(over_limit):.1f} tokens")
    
    # Length buckets
    print("\n" + "=" * 60)
    print("LENGTH DISTRIBUTION (combined)")
    print("=" * 60)
    
    buckets = [
        (0, 128, "0-128"),
        (129, 256, "129-256"),
        (257, 384, "257-384"),
        (385, 512, "385-512"),
        (513, 768, "513-768 (truncated)"),
        (769, 1024, "769-1024 (truncated)"),
        (1025, float('inf'), "1025+ (truncated)")
    ]
    
    for low, high, label in buckets:
        count = sum(1 for l in combined_lengths if low <= l <= high)
        pct = count / len(combined_lengths) * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:25} {count:6,} ({pct:5.1f}%) {bar}")
    
    # Sample long examples
    print("\n" + "=" * 60)
    print("SAMPLE TRUNCATED EXAMPLES (5 longest)")
    print("=" * 60)
    
    indexed = [(i, l) for i, l in enumerate(combined_lengths)]
    indexed.sort(key=lambda x: -x[1])
    
    for idx, length in indexed[:5]:
        q, a = qa_pairs[idx]
        print(f"\n[{length} tokens - {length - MAX_SEQ_LEN} over limit]")
        print(f"Q: {q[:100]}...")
        print(f"A: {a[:200]}...")
    
    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if truncated_count / len(qa_pairs) > 0.2:
        print("\n⚠️  More than 20% of data is truncated!")
        print("   Consider:")
        print("   - Increasing max_seq_len to 768 or 1024")
        print("   - Filtering out very long examples")
        print("   - Truncating answers during preprocessing")
    elif truncated_count / len(qa_pairs) > 0.05:
        print("\n⚠️  5-20% of data is truncated.")
        print("   Consider increasing max_seq_len to 768")
    else:
        print("\n✓ Less than 5% truncation. Current max_seq_len is acceptable.")
    
    return 0


if __name__ == "__main__":
    exit(main())
