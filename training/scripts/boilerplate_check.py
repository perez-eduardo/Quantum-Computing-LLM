#!/usr/bin/env python3
"""
Boilerplate Detection for Model v3
Checks evaluation outputs for known garbage phrases from v1 ChatGPT data.
"""

import json
import re
from pathlib import Path

# Known boilerplate phrases from v1 ChatGPT garbage data
BOILERPLATE_PHRASES = [
    "in fault-tolerant settings",
    "in fault tolerant settings", 
    "fault-tolerant quantum",
    "in practice, this",
    "in real-world applications",
    "it is important to note",
    "it's important to note",
    "this is because",
    "this means that",
    "in other words",
    "for example, if",
    "as mentioned earlier",
    "as we discussed",
    "to put it simply",
    "in summary,",
    "to summarize,",
    "in conclusion,",
    "generally speaking",
    "broadly speaking",
    "technically speaking",
]

# Template patterns (questions that were duplicated with only numbers changed)
TEMPLATE_PATTERNS = [
    r"how many basis states for \d+ qubits",
    r"how many qubits for \d+ basis states", 
    r"what is 2\^\d+",
    r"calculate 2 to the power of \d+",
]

def load_results(path):
    """Load evaluation results JSON."""
    with open(path) as f:
        return json.load(f)

def check_boilerplate(text, phrases):
    """Check if text contains any boilerplate phrases."""
    text_lower = text.lower()
    found = []
    for phrase in phrases:
        if phrase.lower() in text_lower:
            found.append(phrase)
    return found

def check_templates(text, patterns):
    """Check if text matches template patterns."""
    text_lower = text.lower()
    found = []
    for pattern in patterns:
        if re.search(pattern, text_lower):
            found.append(pattern)
    return found

def main():
    results_path = Path("scripts/evaluation_results.json")
    
    if not results_path.exists():
        print(f"ERROR: {results_path} not found")
        print("Run evaluate_model.py first to generate results.")
        return
    
    print("=" * 60)
    print("BOILERPLATE DETECTION: Model v3 Outputs")
    print("=" * 60)
    
    results = load_results(results_path)
    detailed = results.get("detailed_results", [])
    
    total = len(detailed)
    boilerplate_count = 0
    template_count = 0
    
    boilerplate_examples = []
    template_examples = []
    
    for item in detailed:
        answer = item.get("answer", "")
        question = item.get("question", "")
        
        # Check answer for boilerplate
        bp_found = check_boilerplate(answer, BOILERPLATE_PHRASES)
        if bp_found:
            boilerplate_count += 1
            boilerplate_examples.append({
                "id": item["id"],
                "question": question[:50],
                "phrases": bp_found
            })
        
        # Check question for template patterns
        tp_found = check_templates(question, TEMPLATE_PATTERNS)
        if tp_found:
            template_count += 1
            template_examples.append({
                "id": item["id"],
                "question": question[:50],
                "patterns": tp_found
            })
    
    # Summary
    print(f"\nTotal responses analyzed: {total}")
    print()
    
    print("BOILERPLATE PHRASES:")
    print(f"  Contaminated: {boilerplate_count}/{total} ({100*boilerplate_count/total:.1f}%)")
    if boilerplate_examples:
        print("  Examples:")
        for ex in boilerplate_examples[:5]:
            print(f"    [{ex['id']}] {ex['question']}...")
            print(f"         Found: {ex['phrases']}")
    else:
        print("  None detected!")
    
    print()
    print("TEMPLATE PATTERNS:")
    print(f"  Detected: {template_count}/{total} ({100*template_count/total:.1f}%)")
    if template_examples:
        print("  Examples:")
        for ex in template_examples[:5]:
            print(f"    [{ex['id']}] {ex['question']}...")
    else:
        print("  None detected!")
    
    # Comparison with v1
    print()
    print("=" * 60)
    print("COMPARISON WITH V1")
    print("=" * 60)
    print()
    print("| Metric              | v1 (Garbage) | v3 (Clean) |")
    print("|---------------------|--------------|------------|")
    print(f"| Boilerplate phrases | 83.4%        | {100*boilerplate_count/total:.1f}%        |")
    print(f"| Template patterns   | 59.0%        | {100*template_count/total:.1f}%        |")
    print()
    
    if boilerplate_count == 0 and template_count == 0:
        print("SUCCESS: No contamination detected in v3 outputs!")
    else:
        print("WARNING: Some contamination still present.")

if __name__ == "__main__":
    main()
