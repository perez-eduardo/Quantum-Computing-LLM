"""
Inspect Short Examples

88% of training data is under 128 tokens.
Are these high quality or garbage?

Run on HPC.
"""

import csv
import random
from pathlib import Path

DATA_PATH = Path("data/combined_qa_final.csv")
TOKENIZER_PATH = Path("tokenizer.json")


def main():
    print("=" * 60)
    print("SHORT EXAMPLES INSPECTION")
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
    
    # Categorize by length
    print("\nCategorizing by token length...")
    
    very_short = []   # < 50 tokens
    short = []        # 50-128 tokens
    medium = []       # 128-256 tokens
    long_examples = [] # > 256 tokens
    
    for row in rows:
        q = row.get('question', '')
        a = row.get('answer', '')
        source = row.get('source', 'unknown')
        
        formatted = f"Q: {q}\nA: {a}"
        tokens = len(tokenizer.encode(formatted).ids)
        
        item = {
            'question': q,
            'answer': a,
            'source': source,
            'tokens': tokens
        }
        
        if tokens < 50:
            very_short.append(item)
        elif tokens < 128:
            short.append(item)
        elif tokens < 256:
            medium.append(item)
        else:
            long_examples.append(item)
    
    print(f"  Very short (<50 tokens): {len(very_short):,} ({len(very_short)/len(rows)*100:.1f}%)")
    print(f"  Short (50-128 tokens): {len(short):,} ({len(short)/len(rows)*100:.1f}%)")
    print(f"  Medium (128-256 tokens): {len(medium):,} ({len(medium)/len(rows)*100:.1f}%)")
    print(f"  Long (>256 tokens): {len(long_examples):,} ({len(long_examples)/len(rows)*100:.1f}%)")
    
    # Sample very short examples
    print("\n" + "=" * 60)
    print("VERY SHORT EXAMPLES (<50 tokens) - Random Sample")
    print("=" * 60)
    
    random.seed(42)
    for i, item in enumerate(random.sample(very_short, min(10, len(very_short))), 1):
        print(f"\n[{i}] {item['tokens']} tokens - {item['source']}")
        print(f"Q: {item['question']}")
        print(f"A: {item['answer']}")
    
    # Sample short examples
    print("\n" + "=" * 60)
    print("SHORT EXAMPLES (50-128 tokens) - Random Sample")
    print("=" * 60)
    
    for i, item in enumerate(random.sample(short, min(10, len(short))), 1):
        print(f"\n[{i}] {item['tokens']} tokens - {item['source']}")
        print(f"Q: {item['question']}")
        print(f"A: {item['answer'][:300]}...")
    
    # Check answer length distribution in short examples
    print("\n" + "=" * 60)
    print("ANSWER LENGTH IN SHORT EXAMPLES")
    print("=" * 60)
    
    very_short_answers = [len(item['answer']) for item in very_short]
    short_answers = [len(item['answer']) for item in short]
    
    print(f"\nVery short examples (<50 tokens):")
    print(f"  Answer char length: min={min(very_short_answers)}, max={max(very_short_answers)}, median={sorted(very_short_answers)[len(very_short_answers)//2]}")
    
    print(f"\nShort examples (50-128 tokens):")
    print(f"  Answer char length: min={min(short_answers)}, max={max(short_answers)}, median={sorted(short_answers)[len(short_answers)//2]}")
    
    # Source breakdown for short examples
    print("\n" + "=" * 60)
    print("SOURCE BREAKDOWN")
    print("=" * 60)
    
    for name, group in [("Very short", very_short), ("Short", short)]:
        print(f"\n{name}:")
        sources = {}
        for item in group:
            src = item['source']
            sources[src] = sources.get(src, 0) + 1
        for src, count in sorted(sources.items(), key=lambda x: -x[1]):
            print(f"  {src}: {count:,} ({count/len(group)*100:.1f}%)")
    
    # Quality check: do short answers actually answer the question?
    print("\n" + "=" * 60)
    print("QUALITY INDICATORS")
    print("=" * 60)
    
    # Check for question marks in answers (might indicate non-answers)
    questions_in_answers = sum(1 for item in very_short + short if '?' in item['answer'])
    print(f"  Short examples with '?' in answer: {questions_in_answers}")
    
    # Check for very short answers
    tiny_answers = sum(1 for item in very_short + short if len(item['answer']) < 50)
    print(f"  Short examples with <50 char answer: {tiny_answers}")
    
    # Check for code in short examples
    code_indicators = ['```', 'import ', 'def ', 'print(']
    code_in_short = sum(1 for item in very_short + short if any(ind in item['answer'] for ind in code_indicators))
    print(f"  Short examples with code: {code_in_short}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    total_short = len(very_short) + len(short)
    if tiny_answers / total_short > 0.1:
        print(f"\n⚠️  {tiny_answers/total_short*100:.0f}% have very short answers.")
        print("   Consider filtering examples with <50 char answers.")
    else:
        print("\n✓ Short examples appear to have substantive answers.")
    
    return 0


if __name__ == "__main__":
    exit(main())
