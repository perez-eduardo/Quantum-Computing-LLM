"""
Verify Dataset v2

Comprehensive verification of the final training dataset.
Checks Q&A data, book data, and tokenizer.

Run on HPC:
    cd ~/hpc-share/quantum-llm
    source venv/bin/activate
    python scripts/verify_dataset_v2.py
"""

import pandas as pd
import re
from pathlib import Path
from collections import Counter
from tokenizers import Tokenizer


def verify_qa_data(df: pd.DataFrame) -> dict:
    """Verify Q&A dataset quality."""
    issues = []
    stats = {}
    
    # Basic counts
    stats['total_rows'] = len(df)
    
    # Null/empty checks
    null_q = df['question'].isna().sum()
    null_a = df['answer'].isna().sum()
    empty_q = (df['question'].astype(str).str.strip() == '').sum()
    empty_a = (df['answer'].astype(str).str.strip() == '').sum()
    
    stats['null_questions'] = null_q
    stats['null_answers'] = null_a
    stats['empty_questions'] = empty_q
    stats['empty_answers'] = empty_a
    
    if null_q > 0 or null_a > 0:
        issues.append(f"Found {null_q} null questions, {null_a} null answers")
    if empty_q > 0 or empty_a > 0:
        issues.append(f"Found {empty_q} empty questions, {empty_a} empty answers")
    
    # Length stats
    df['q_len'] = df['question'].astype(str).str.len()
    df['a_len'] = df['answer'].astype(str).str.len()
    
    stats['q_len_min'] = df['q_len'].min()
    stats['q_len_max'] = df['q_len'].max()
    stats['q_len_median'] = df['q_len'].median()
    stats['a_len_min'] = df['a_len'].min()
    stats['a_len_max'] = df['a_len'].max()
    stats['a_len_median'] = df['a_len'].median()
    
    # Very short
    short_q = (df['q_len'] < 10).sum()
    short_a = (df['a_len'] < 20).sum()
    stats['very_short_questions'] = short_q
    stats['very_short_answers'] = short_a
    
    if short_q > 0:
        issues.append(f"Found {short_q} very short questions (<10 chars)")
    if short_a > 0:
        issues.append(f"Found {short_a} very short answers (<20 chars)")
    
    # Duplicates
    dup_q = df['question'].duplicated().sum()
    dup_a = df['answer'].duplicated().sum()
    stats['duplicate_questions'] = dup_q
    stats['duplicate_answers'] = dup_a
    
    if dup_q > 0:
        issues.append(f"Found {dup_q} duplicate questions")
    if dup_a > 0:
        issues.append(f"Found {dup_a} duplicate answers")
    
    # Source distribution
    if 'source' in df.columns:
        stats['source_distribution'] = df['source'].value_counts().to_dict()
    
    # HTML entities check
    html_pattern = r'&[a-zA-Z]+;|&#\d+;'
    html_q = df['question'].astype(str).str.contains(html_pattern, regex=True).sum()
    html_a = df['answer'].astype(str).str.contains(html_pattern, regex=True).sum()
    stats['html_entities'] = html_q + html_a
    
    if html_q + html_a > 100:  # Allow some (might be intentional)
        issues.append(f"Found {html_q + html_a} rows with HTML entities")
    
    # Clean up temp columns
    df.drop(columns=['q_len', 'a_len'], inplace=True, errors='ignore')
    
    return {'stats': stats, 'issues': issues}


def verify_book_data(book_path: Path) -> dict:
    """Verify book text quality."""
    issues = []
    stats = {}
    
    with open(book_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    stats['total_chars'] = len(text)
    stats['total_words'] = len(text.split())
    
    # Paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    stats['paragraphs'] = len(paragraphs)
    
    if len(paragraphs) > 0:
        para_lens = [len(p) for p in paragraphs]
        stats['para_len_min'] = min(para_lens)
        stats['para_len_max'] = max(para_lens)
        stats['para_len_median'] = sorted(para_lens)[len(para_lens)//2]
        
        # Very short paragraphs
        short_paras = sum(1 for p in para_lens if p < 50)
        stats['short_paragraphs'] = short_paras
        
        if short_paras > len(paragraphs) * 0.1:
            issues.append(f"Found {short_paras} very short paragraphs (<50 chars)")
    
    # Control characters
    control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
    stats['control_chars'] = control_chars
    
    if control_chars > 0:
        issues.append(f"Found {control_chars} control characters")
    
    return {'stats': stats, 'issues': issues}


def verify_tokenizer(tokenizer_path: Path, qa_path: Path) -> dict:
    """Verify tokenizer quality."""
    issues = []
    stats = {}
    
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    stats['vocab_size'] = tokenizer.get_vocab_size()
    
    # Check special tokens
    vocab = tokenizer.get_vocab()
    expected = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    
    for token, expected_id in expected.items():
        actual_id = vocab.get(token, -1)
        if actual_id != expected_id:
            issues.append(f"Special token {token} has wrong ID: {actual_id} (expected {expected_id})")
    
    stats['special_tokens_ok'] = len(issues) == 0
    
    # Test encoding/decoding
    test_sentences = [
        "What is a qubit?",
        "Quantum entanglement is fascinating.",
        "The Hadamard gate creates superposition.",
    ]
    
    decode_ok = True
    for sent in test_sentences:
        encoded = tokenizer.encode(sent)
        decoded = tokenizer.decode(encoded.ids)
        if sent.lower() not in decoded.lower():
            decode_ok = False
            issues.append(f"Decode mismatch: '{sent}' -> '{decoded}'")
    
    stats['encoding_ok'] = decode_ok
    
    # Token length distribution on sample
    df = pd.read_csv(qa_path, nrows=1000)
    token_counts = []
    for _, row in df.iterrows():
        formatted = f"Q: {row['question']}\nA: {row['answer']}"
        token_counts.append(len(tokenizer.encode(formatted).ids))
    
    stats['sample_token_mean'] = sum(token_counts) / len(token_counts)
    stats['sample_token_max'] = max(token_counts)
    
    return {'stats': stats, 'issues': issues}


def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    data_dir = Path.home() / "hpc-share/quantum-llm/data"
    tokenizer_path = Path.home() / "hpc-share/quantum-llm/tokenizer.json"
    
    qa_path = data_dir / "combined_qa_filtered.csv"
    book_path = data_dir / "combined_books.txt"
    
    print("=" * 60)
    print("VERIFY DATASET v2")
    print("=" * 60)
    print("\nComprehensive verification of training data.")
    
    all_issues = []
    
    # Check files exist
    print_section("FILE CHECK")
    files = [qa_path, book_path, tokenizer_path]
    for f in files:
        exists = "OK" if f.exists() else "MISSING"
        print(f"  {f.name}: {exists}")
        if not f.exists():
            all_issues.append(f"Missing file: {f}")
    
    if any(not f.exists() for f in files):
        print("\nERROR: Missing required files. Run previous scripts first.")
        return 1
    
    # Verify Q&A data
    print_section("Q&A DATA VERIFICATION")
    df = pd.read_csv(qa_path)
    qa_result = verify_qa_data(df)
    
    print("\n  Stats:")
    for key, value in qa_result['stats'].items():
        if isinstance(value, dict):
            print(f"    {key}:")
            for k, v in value.items():
                print(f"      {k}: {v:,}")
        else:
            print(f"    {key}: {value:,}" if isinstance(value, (int, float)) else f"    {key}: {value}")
    
    if qa_result['issues']:
        print("\n  Issues:")
        for issue in qa_result['issues']:
            print(f"    - {issue}")
        all_issues.extend(qa_result['issues'])
    else:
        print("\n  No issues found.")
    
    # Verify book data
    print_section("BOOK DATA VERIFICATION")
    book_result = verify_book_data(book_path)
    
    print("\n  Stats:")
    for key, value in book_result['stats'].items():
        print(f"    {key}: {value:,}" if isinstance(value, (int, float)) else f"    {key}: {value}")
    
    if book_result['issues']:
        print("\n  Issues:")
        for issue in book_result['issues']:
            print(f"    - {issue}")
        all_issues.extend(book_result['issues'])
    else:
        print("\n  No issues found.")
    
    # Verify tokenizer
    print_section("TOKENIZER VERIFICATION")
    tok_result = verify_tokenizer(tokenizer_path, qa_path)
    
    print("\n  Stats:")
    for key, value in tok_result['stats'].items():
        if isinstance(value, float):
            print(f"    {key}: {value:.1f}")
        else:
            print(f"    {key}: {value}")
    
    if tok_result['issues']:
        print("\n  Issues:")
        for issue in tok_result['issues']:
            print(f"    - {issue}")
        all_issues.extend(tok_result['issues'])
    else:
        print("\n  No issues found.")
    
    # Sample Q&A pairs
    print_section("SAMPLE Q&A PAIRS (5 random)")
    samples = df.sample(5, random_state=42)
    for i, (_, row) in enumerate(samples.iterrows(), 1):
        q = str(row['question'])[:80]
        a = str(row['answer'])[:120]
        src = row.get('source', 'unknown')
        print(f"\n  [{i}] ({src})")
        print(f"  Q: {q}...")
        print(f"  A: {a}...")
    
    # Final summary
    print_section("FINAL SUMMARY")
    
    print(f"\n  Q&A pairs:     {len(df):,}")
    print(f"  Book words:    {book_result['stats']['total_words']:,}")
    print(f"  Vocab size:    {tok_result['stats']['vocab_size']:,}")
    
    if all_issues:
        print(f"\n  ISSUES FOUND: {len(all_issues)}")
        for issue in all_issues:
            print(f"    - {issue}")
        print("\n  Review issues before training!")
    else:
        print("\n  ALL CHECKS PASSED")
        print("\n  Ready for training!")
    
    # Estimate tokens
    print_section("TOKEN ESTIMATE")
    
    # Q&A tokens (rough estimate)
    qa_chars = df['question'].astype(str).str.len().sum() + df['answer'].astype(str).str.len().sum()
    qa_tokens_est = qa_chars / 4  # ~4 chars per token
    
    # Book tokens
    book_tokens_est = book_result['stats']['total_chars'] / 4
    
    total_tokens = qa_tokens_est + book_tokens_est
    
    print(f"\n  Q&A tokens (est):    ~{qa_tokens_est/1e6:.1f}M")
    print(f"  Book tokens (est):   ~{book_tokens_est/1e6:.1f}M")
    print(f"  Total (est):         ~{total_tokens/1e6:.1f}M")
    print(f"\n  Token:param ratio:   ~{total_tokens/1.2e6:.1f}:1 (1.2M model)")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    if not all_issues:
        print("""
  1. Update dataset.py if needed for new file names
  2. Run training:
     
     sbatch train_job_v2.sh
     
  Or manually:
     
     python scripts/train.py \\
         --qa_path ../data/combined_qa_filtered.csv \\
         --book_path ../data/combined_books.txt \\
         --tokenizer_path ../tokenizer.json \\
         --epochs 10 \\
         --min_lr 1e-5 \\
         --warmup_ratio 0.05
""")
    
    return 0 if not all_issues else 1


if __name__ == "__main__":
    exit(main())
