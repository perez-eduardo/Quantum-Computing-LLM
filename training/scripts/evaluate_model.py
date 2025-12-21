"""
Structured evaluation: Score model against 50-question test set.
Run from: ~/hpc-share/quantum-llm/scripts/

Outputs:
  - Per-question scores
  - Category breakdown
  - Difficulty breakdown
  - Overall metrics
  - Saves results to evaluation_results.json
"""

import torch
import json
import sys
import re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from model import QuantumLLM

# Paths
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "model" / "final_model.pt"
CONFIG_PATH = BASE_DIR / "model" / "config.json"
TOKENIZER_PATH = BASE_DIR / "tokenizer.json"
TEST_SET_PATH = Path(__file__).parent / "test_questions.json"
OUTPUT_PATH = Path(__file__).parent / "evaluation_results.json"


def load_tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


def extract_answer(response):
    if "Answer:" in response:
        return response.split("Answer:")[-1].strip()
    return response.strip()


def score_response(answer, expected_keywords):
    """
    Score response based on keyword presence.
    Returns (score, matched_keywords, missing_keywords)
    """
    answer_lower = answer.lower()
    
    matched = []
    missing = []
    
    for keyword in expected_keywords:
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        if re.search(pattern, answer_lower):
            matched.append(keyword)
        else:
            missing.append(keyword)
    
    score = len(matched) / len(expected_keywords) if expected_keywords else 0
    return score, matched, missing


def check_quality_flags(answer):
    """Check for quality issues in the answer."""
    flags = []
    
    if len(answer.strip()) < 10:
        flags.append("too_short")
    
    words = answer.split()
    if len(words) > 10:
        for i in range(len(words) - 5):
            phrase = " ".join(words[i:i+3])
            if answer.count(phrase) > 2:
                flags.append("repetitive")
                break
    
    alpha_chars = sum(1 for c in answer if c.isalpha())
    if len(answer) > 20 and alpha_chars / len(answer) < 0.5:
        flags.append("possible_gibberish")
    
    return flags


def main():
    print("=" * 60)
    print("STRUCTURED EVALUATION: 50-Question Test Set")
    print("=" * 60)
    
    # Load test set
    print("\nLoading test set...")
    with open(TEST_SET_PATH) as f:
        test_data = json.load(f)
    questions = test_data["questions"]
    print(f"  Loaded {len(questions)} questions")
    
    # Load config
    print("\nLoading config...")
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    
    # Load model
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    model = QuantumLLM(
        vocab_size=config["vocab_size"],
        dim=config["dim"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        ff_dim=config["ff_dim"],
        max_seq_len=config["max_seq_len"],
        dropout=0.0
    ).to(device)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Run evaluation
    print("\n" + "=" * 60)
    print("RUNNING EVALUATION")
    print("=" * 60)
    
    results = []
    category_scores = {}
    difficulty_scores = {}
    
    for i, q in enumerate(questions, 1):
        prompt = f"Question: {q['question']}\nAnswer:"
        print(f"\n[{i}/{len(questions)}] {q['question'][:50]}...")
        
        # Encode and generate
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], device=device)
        
        output_ids = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            eos_token_id=1
        )
        
        response = tokenizer.decode(output_ids[0].tolist())
        answer = extract_answer(response)
        
        # Score
        score, matched, missing = score_response(answer, q["expected_keywords"])
        flags = check_quality_flags(answer)
        
        result = {
            "id": q["id"],
            "category": q["category"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "answer": answer[:500],
            "expected_keywords": q["expected_keywords"],
            "matched_keywords": matched,
            "missing_keywords": missing,
            "score": score,
            "quality_flags": flags
        }
        results.append(result)
        
        cat = q["category"]
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(score)
        
        diff = q["difficulty"]
        if diff not in difficulty_scores:
            difficulty_scores[diff] = []
        difficulty_scores[diff].append(score)
        
        print(f"  Score: {score:.0%} | Matched: {matched} | Flags: {flags or 'none'}")
    
    # Calculate summaries
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    overall_score = sum(r["score"] for r in results) / len(results)
    print(f"\nOVERALL SCORE: {overall_score:.1%}")
    
    print("\nBY CATEGORY:")
    category_summary = {}
    for cat, scores in sorted(category_scores.items()):
        avg = sum(scores) / len(scores)
        category_summary[cat] = {"average": avg, "count": len(scores)}
        print(f"  {cat:15} {avg:5.1%}  (n={len(scores)})")
    
    print("\nBY DIFFICULTY:")
    difficulty_summary = {}
    for diff in ["easy", "medium", "hard"]:
        if diff in difficulty_scores:
            scores = difficulty_scores[diff]
            avg = sum(scores) / len(scores)
            difficulty_summary[diff] = {"average": avg, "count": len(scores)}
            print(f"  {diff:15} {avg:5.1%}  (n={len(scores)})")
    
    print("\nQUALITY FLAGS:")
    flag_counts = {}
    for r in results:
        for flag in r["quality_flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    if flag_counts:
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            print(f"  {flag:20} {count} occurrences")
    else:
        print("  No quality issues detected")
    
    print("\nWORST PERFORMING (score < 20%):")
    worst = [r for r in results if r["score"] < 0.2]
    for r in sorted(worst, key=lambda x: x["score"])[:5]:
        print(f"  [{r['id']}] {r['question'][:40]}... (score: {r['score']:.0%})")
    
    print("\nBEST PERFORMING (score = 100%):")
    best = [r for r in results if r["score"] == 1.0]
    print(f"  {len(best)} questions with perfect keyword match")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "model_config": config,
        "overall_score": overall_score,
        "category_summary": category_summary,
        "difficulty_summary": difficulty_summary,
        "quality_flag_counts": flag_counts,
        "detailed_results": results
    }
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to: {OUTPUT_PATH}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print("\nInterpretation guide:")
    print("  - 50%+: Model learned relevant vocabulary")
    print("  - 30-50%: Partial keyword coverage, expected for 1.2M params")
    print("  - <30%: Model may need more training or data")
    print("\nRemember: Keyword matching != coherent answers")
    print("RAG will compensate for answer quality.")


if __name__ == "__main__":
    main()
