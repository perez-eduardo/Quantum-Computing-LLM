"""
Clean ChatGPT Training Data - Remove Boilerplate

Strips repetitive boilerplate phrases from answers.
Run locally.

Usage:
    python clean_boilerplate.py
"""

import csv
import re
from pathlib import Path

# Paths
INPUT_PATH = Path("data/raw/chatgpt_qa_final.csv")
OUTPUT_PATH = Path("data/raw/chatgpt_qa_cleaned_v2.csv")

# Boilerplate phrases to remove (order matters - check longer phrases first)
BOILERPLATE_PHRASES = [
    # Full sentences to strip
    "In simulators, phase effects only show up after interference with later gates.",
    "In simulators, two-qubit operations tend to be the main error source.",
    "In simulators, error mitigation may reduce bias but doesn't fully protect quantum information.",
    "In simulators, error mitigation can reduce bias but doesn't fully protect quantum information.",
    "In simulators, measurement statistics require many shots to estimate reliably.",
    "In simulators, noise accumulates with circuit depth.",
    "In simulators, full-state reconstruction scales poorly with qubit count.",
    "In simulators, limited connectivity can force extra SWAP routing.",
    "In simulators, limited connectivity may force extra SWAP routing.",
    "In simulators, compiler choices may noticeably change depth and fidelity.",
    
    "In fault-tolerant settings, phase effects only show up after interference with later gates.",
    "In fault-tolerant settings, two-qubit operations tend to be the main error source.",
    "In fault-tolerant settings, error mitigation may reduce bias but doesn't fully protect quantum information.",
    "In fault-tolerant settings, error mitigation can reduce bias but doesn't fully protect quantum information.",
    "In fault-tolerant settings, measurement statistics require many shots to estimate reliably.",
    "In fault-tolerant settings, noise accumulates with circuit depth.",
    "In fault-tolerant settings, full-state reconstruction scales poorly with qubit count.",
    "In fault-tolerant settings, limited connectivity can force extra SWAP routing.",
    "In fault-tolerant settings, limited connectivity may force extra SWAP routing.",
    "In fault-tolerant settings, compiler choices may noticeably change depth and fidelity.",
    
    "In many tutorials, phase effects only show up after interference with later gates.",
    "In many tutorials, two-qubit operations tend to be the main error source.",
    "In many tutorials, error mitigation may reduce bias but doesn't fully protect quantum information.",
    "In many tutorials, error mitigation can reduce bias but doesn't fully protect quantum information.",
    "In many tutorials, measurement statistics require many shots to estimate reliably.",
    "In many tutorials, noise accumulates with circuit depth.",
    "In many tutorials, full-state reconstruction scales poorly with qubit count.",
    "In many tutorials, limited connectivity can force extra SWAP routing.",
    "In many tutorials, limited connectivity may force extra SWAP routing.",
    "In many tutorials, compiler choices may noticeably change depth and fidelity.",
    
    "In NISQ experiments, phase effects only show up after interference with later gates.",
    "In NISQ experiments, two-qubit operations tend to be the main error source.",
    "In NISQ experiments, error mitigation may reduce bias but doesn't fully protect quantum information.",
    "In NISQ experiments, error mitigation can reduce bias but doesn't fully protect quantum information.",
    "In NISQ experiments, measurement statistics require many shots to estimate reliably.",
    "In NISQ experiments, noise accumulates with circuit depth.",
    "In NISQ experiments, full-state reconstruction scales poorly with qubit count.",
    "In NISQ experiments, limited connectivity can force extra SWAP routing.",
    "In NISQ experiments, limited connectivity may force extra SWAP routing.",
    "In NISQ experiments, compiler choices may noticeably change depth and fidelity.",
    
    "In practical workflows, phase effects only show up after interference with later gates.",
    "In practical workflows, two-qubit operations tend to be the main error source.",
    "In practical workflows, error mitigation may reduce bias but doesn't fully protect quantum information.",
    "In practical workflows, error mitigation can reduce bias but doesn't fully protect quantum information.",
    "In practical workflows, measurement statistics require many shots to estimate reliably.",
    "In practical workflows, noise accumulates with circuit depth.",
    "In practical workflows, full-state reconstruction scales poorly with qubit count.",
    "In practical workflows, limited connectivity can force extra SWAP routing.",
    "In practical workflows, limited connectivity may force extra SWAP routing.",
    "In practical workflows, compiler choices may noticeably change depth and fidelity.",
    
    "On real devices, phase effects only show up after interference with later gates.",
    "On real devices, two-qubit operations tend to be the main error source.",
    "On real devices, error mitigation may reduce bias but doesn't fully protect quantum information.",
    "On real devices, error mitigation can reduce bias but doesn't fully protect quantum information.",
    "On real devices, measurement statistics require many shots to estimate reliably.",
    "On real devices, noise accumulates with circuit depth.",
    "On real devices, full-state reconstruction scales poorly with qubit count.",
    "On real devices, limited connectivity can force extra SWAP routing.",
    "On real devices, limited connectivity may force extra SWAP routing.",
    "On real devices, compiler choices may noticeably change depth and fidelity.",
    
    "Remember that probabilities come from |amplitude|², not the amplitudes directly.",
    "Be careful: different measurement bases can change what information you actually access.",
    "Don't confuse this with classical parallelism—you can't read out all branches of a superposition.",
]

# Regex patterns for variable boilerplate
BOILERPLATE_PATTERNS = [
    r"In simulators,[^.]+\.",
    r"In fault-tolerant settings,[^.]+\.",
    r"In many tutorials,[^.]+\.",
    r"In NISQ experiments,[^.]+\.",
    r"In practical workflows,[^.]+\.",
    r"On real devices,[^.]+\.",
    r"\(In context:[^)]+\.\.\)\.",
    r"\(In context:[^)]+\)",
]


def clean_answer(answer: str) -> str:
    """Remove boilerplate from answer."""
    if not answer:
        return answer
    
    cleaned = answer.strip()
    
    # Remove exact phrase matches
    for phrase in BOILERPLATE_PHRASES:
        cleaned = cleaned.replace(phrase, "")
    
    # Remove pattern matches
    for pattern in BOILERPLATE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)
    
    # Clean up whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\s+\.', '.', cleaned)
    cleaned = re.sub(r'\.\s*\.', '.', cleaned)
    cleaned = cleaned.strip()
    
    # Remove trailing incomplete sentences (if answer now ends mid-sentence)
    # Keep only complete sentences
    if cleaned and not cleaned.endswith(('.', '?', '!', '"', ')')):
        # Find last sentence end
        last_period = max(cleaned.rfind('.'), cleaned.rfind('?'), cleaned.rfind('!'))
        if last_period > len(cleaned) * 0.5:  # Only trim if we keep most of answer
            cleaned = cleaned[:last_period + 1]
    
    return cleaned


def main():
    print("=" * 60)
    print("CLEANING BOILERPLATE FROM CHATGPT DATA")
    print("=" * 60)
    
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} not found")
        return 1
    
    # Load data
    print(f"\nLoading {INPUT_PATH}...")
    rows = []
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    print(f"  Loaded {len(rows):,} rows")
    
    # Detect column names
    q_col = fieldnames[0]
    a_col = fieldnames[1]
    print(f"  Question column: {q_col}")
    print(f"  Answer column: {a_col}")
    
    # Clean answers
    print("\nCleaning answers...")
    
    cleaned_count = 0
    removed_count = 0
    cleaned_rows = []
    
    for row in rows:
        original = row[a_col]
        cleaned = clean_answer(original)
        
        # Skip if answer is now too short
        if len(cleaned) < 50:
            removed_count += 1
            continue
        
        if cleaned != original:
            cleaned_count += 1
        
        row[a_col] = cleaned
        cleaned_rows.append(row)
    
    print(f"  Cleaned: {cleaned_count:,} answers modified")
    print(f"  Removed: {removed_count:,} answers (too short after cleaning)")
    print(f"  Kept: {len(cleaned_rows):,} answers")
    
    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)
    
    print(f"  Saved {len(cleaned_rows):,} rows")
    
    # Verify
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    # Re-check boilerplate
    boilerplate_check = [
        "In fault-tolerant settings",
        "In practical workflows",
        "In NISQ experiments",
        "In many tutorials",
        "In simulators",
        "On real devices",
    ]
    
    remaining = 0
    for row in cleaned_rows:
        answer = row[a_col].lower()
        if any(phrase.lower() in answer for phrase in boilerplate_check):
            remaining += 1
    
    print(f"  Answers still with boilerplate: {remaining:,} ({remaining/len(cleaned_rows)*100:.1f}%)")
    
    # Show samples
    print("\n" + "=" * 60)
    print("SAMPLE CLEANED ANSWERS")
    print("=" * 60)
    
    import random
    random.seed(42)
    samples = random.sample(cleaned_rows, min(5, len(cleaned_rows)))
    
    for i, row in enumerate(samples, 1):
        print(f"\n[Sample {i}]")
        print(f"Q: {row[q_col][:100]}...")
        print(f"A: {row[a_col][:300]}...")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Review chatgpt_qa_cleaned_v2.csv")
    print(f"  2. Re-run combine_qa.py with cleaned data")
    print(f"  3. Upload new combined_qa_final.csv to HPC")
    print(f"  4. Retrain model")
    
    return 0


if __name__ == "__main__":
    exit(main())
