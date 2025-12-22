"""
Inspect ChatGPT Training Data for Boilerplate

Finds repetitive phrases that the model memorized.
Run locally.

Usage:
    python inspect_boilerplate.py
"""

import csv
from pathlib import Path
from collections import Counter

# Path to ChatGPT data
DATA_PATH = Path("data/raw/chatgpt_qa_final.csv")

# Boilerplate phrases found in model outputs
BOILERPLATE_PHRASES = [
    "In fault-tolerant settings",
    "In practical workflows",
    "In NISQ experiments",
    "In many tutorials",
    "In simulators",
    "On real devices",
    "Don't confuse this with classical parallelism",
    "error mitigation may reduce bias",
    "error mitigation can reduce bias",
    "doesn't fully protect quantum information",
    "phase effects only show up after interference",
    "two-qubit operations tend to be the main error source",
    "measurement statistics require many shots",
    "limited connectivity can force extra SWAP",
    "limited connectivity may force extra SWAP",
    "compiler choices may noticeably change depth",
    "noise accumulates with circuit depth",
    "full-state reconstruction scales poorly",
    "Remember that probabilities come from |amplitude|²",
    "(In context:",
]


def main():
    print("=" * 60)
    print("BOILERPLATE INSPECTION")
    print("=" * 60)
    
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found")
        return 1
    
    # Load data
    print(f"\nLoading {DATA_PATH}...")
    answers = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Try different column names
            answer = row.get('answer') or row.get('Answer') or row.get(list(row.keys())[1])
            if answer:
                answers.append(answer)
    
    print(f"  Loaded {len(answers):,} answers")
    
    # Count boilerplate
    print("\n" + "=" * 60)
    print("BOILERPLATE PHRASE COUNTS")
    print("=" * 60)
    
    phrase_counts = {}
    phrase_examples = {}
    
    for phrase in BOILERPLATE_PHRASES:
        count = sum(1 for a in answers if phrase.lower() in a.lower())
        phrase_counts[phrase] = count
        
        # Get example
        for a in answers:
            if phrase.lower() in a.lower():
                phrase_examples[phrase] = a
                break
    
    # Sort by count
    sorted_phrases = sorted(phrase_counts.items(), key=lambda x: -x[1])
    
    total_with_boilerplate = 0
    for phrase, count in sorted_phrases:
        pct = count / len(answers) * 100
        if count > 0:
            total_with_boilerplate += count
        print(f"  {count:6,} ({pct:5.1f}%) | {phrase[:50]}")
    
    # Check how many answers have ANY boilerplate
    print("\n" + "=" * 60)
    print("ANSWERS WITH ANY BOILERPLATE")
    print("=" * 60)
    
    answers_with_any = 0
    answers_with_multiple = 0
    
    for a in answers:
        a_lower = a.lower()
        matches = sum(1 for phrase in BOILERPLATE_PHRASES if phrase.lower() in a_lower)
        if matches > 0:
            answers_with_any += 1
        if matches > 1:
            answers_with_multiple += 1
    
    print(f"  Answers with 1+ boilerplate phrase: {answers_with_any:,} ({answers_with_any/len(answers)*100:.1f}%)")
    print(f"  Answers with 2+ boilerplate phrases: {answers_with_multiple:,} ({answers_with_multiple/len(answers)*100:.1f}%)")
    
    # Show examples
    print("\n" + "=" * 60)
    print("SAMPLE ANSWERS WITH BOILERPLATE")
    print("=" * 60)
    
    shown = 0
    for a in answers:
        a_lower = a.lower()
        matches = [p for p in BOILERPLATE_PHRASES if p.lower() in a_lower]
        if len(matches) >= 2:
            print(f"\n[{len(matches)} phrases matched]")
            print(f"Answer: {a[:500]}...")
            print(f"Matched: {matches}")
            shown += 1
            if shown >= 5:
                break
    
    # Find pattern: answers ending with boilerplate
    print("\n" + "=" * 60)
    print("ANSWERS ENDING WITH BOILERPLATE")
    print("=" * 60)
    
    ending_patterns = [
        "error source.",
        "quantum information.",
        "later gates.",
        "estimate reliably.",
        "circuit depth.",
        "qubit count.",
        "increase depth.",
    ]
    
    for pattern in ending_patterns:
        count = sum(1 for a in answers if a.strip().lower().endswith(pattern.lower()))
        if count > 0:
            print(f"  {count:6,} answers end with '...{pattern}'")
    
    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    if answers_with_any / len(answers) > 0.3:
        print(f"\n⚠️  {answers_with_any/len(answers)*100:.0f}% of answers contain boilerplate!")
        print("   This explains why the model outputs repetitive phrases.")
        print("\n   Options:")
        print("   1. Remove boilerplate phrases from answers")
        print("   2. Filter out answers with boilerplate entirely")
        print("   3. Accept it and rely on RAG to override")
    else:
        print(f"\n   Only {answers_with_any/len(answers)*100:.0f}% have boilerplate. May not be critical.")
    
    return 0


if __name__ == "__main__":
    exit(main())
