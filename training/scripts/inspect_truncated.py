"""
Inspect Truncated Examples

Shows what the 5.5% of examples that exceed 512 tokens look like.
Helps decide if they should be filtered out.

Run locally.
"""

import csv
from pathlib import Path

DATA_PATH = Path("data/raw/combined_qa_final_v2.csv")
TOKENIZER_PATH = Path("training/tokenizer/tokenizer.json")
MAX_SEQ_LEN = 512


def main():
    print("=" * 60)
    print("TRUNCATED EXAMPLES INSPECTION")
    print("=" * 60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    
    # Load data
    print(f"Loading {DATA_PATH}...")
    rows = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"  Loaded {len(rows):,} rows")
    
    # Find truncated examples
    print(f"\nFinding examples that exceed {MAX_SEQ_LEN} tokens...")
    
    truncated = []
    for row in rows:
        q = row.get('question', '')
        a = row.get('answer', '')
        source = row.get('source', 'unknown')
        
        formatted = f"Q: {q}\nA: {a}"
        tokens = len(tokenizer.encode(formatted).ids) + 1  # +1 for EOS
        
        if tokens > MAX_SEQ_LEN:
            truncated.append({
                'question': q,
                'answer': a,
                'source': source,
                'tokens': tokens,
                'over_by': tokens - MAX_SEQ_LEN
            })
    
    print(f"  Found {len(truncated):,} truncated examples ({len(truncated)/len(rows)*100:.1f}%)")
    
    # Breakdown by source
    print("\n" + "=" * 60)
    print("TRUNCATED BY SOURCE")
    print("=" * 60)
    
    by_source = {}
    for t in truncated:
        src = t['source']
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(t)
    
    for src, items in sorted(by_source.items(), key=lambda x: -len(x[1])):
        print(f"  {src}: {len(items):,} ({len(items)/len(truncated)*100:.1f}%)")
    
    # How much are they over?
    print("\n" + "=" * 60)
    print("HOW MUCH OVER THE LIMIT")
    print("=" * 60)
    
    over_amounts = [t['over_by'] for t in truncated]
    buckets = [
        (1, 100, "1-100 tokens over"),
        (101, 500, "101-500 tokens over"),
        (501, 1000, "501-1000 tokens over"),
        (1001, 5000, "1001-5000 tokens over"),
        (5001, float('inf'), "5000+ tokens over"),
    ]
    
    for low, high, label in buckets:
        count = sum(1 for o in over_amounts if low <= o <= high)
        print(f"  {label:25} {count:5,} ({count/len(truncated)*100:.1f}%)")
    
    # Sample truncated examples
    print("\n" + "=" * 60)
    print("SAMPLE TRUNCATED EXAMPLES")
    print("=" * 60)
    
    # Sort by tokens (show worst first)
    truncated_sorted = sorted(truncated, key=lambda x: -x['tokens'])
    
    for i, t in enumerate(truncated_sorted[:10], 1):
        print(f"\n[{i}] {t['tokens']} tokens ({t['over_by']} over) - {t['source']}")
        print(f"Q: {t['question'][:150]}...")
        print(f"A: {t['answer'][:300]}...")
        
        # Show what would be cut
        formatted = f"Q: {t['question']}\nA: {t['answer']}"
        full_tokens = tokenizer.encode(formatted).ids
        kept = tokenizer.decode(full_tokens[:MAX_SEQ_LEN])
        cut = tokenizer.decode(full_tokens[MAX_SEQ_LEN:])
        
        print(f"--- KEPT ({MAX_SEQ_LEN} tokens) ---")
        print(kept[-200:] + "...")
        print(f"--- CUT ({len(full_tokens) - MAX_SEQ_LEN} tokens) ---")
        print(cut[:200] + "...")
    
    # Recommendation
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    # Check if truncated examples are mostly code
    code_indicators = ['```', 'import ', 'def ', 'class ', 'function', '.py', 'print(']
    code_count = 0
    for t in truncated:
        if any(ind in t['answer'] for ind in code_indicators):
            code_count += 1
    
    print(f"  Truncated examples with code: {code_count}/{len(truncated)} ({code_count/len(truncated)*100:.1f}%)")
    
    # Check average answer quality in truncated
    avg_over = sum(t['over_by'] for t in truncated) / len(truncated)
    print(f"  Average tokens over limit: {avg_over:.0f}")
    
    severely_truncated = sum(1 for t in truncated if t['over_by'] > 500)
    print(f"  Severely truncated (>500 over): {severely_truncated}/{len(truncated)} ({severely_truncated/len(truncated)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    if severely_truncated / len(truncated) > 0.5:
        print("\n  Most truncated examples lose significant content.")
        print("  Consider filtering out examples >1024 tokens before truncation.")
    else:
        print("\n  Most truncated examples only lose a small amount.")
        print("  Current truncation may be acceptable.")
    
    return 0


if __name__ == "__main__":
    exit(main())
