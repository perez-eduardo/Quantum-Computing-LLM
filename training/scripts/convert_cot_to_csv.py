"""
Convert CoT_Reasoning_Quantum_Physics_And_Computing.json to CSV format.
Output: question and answer columns only (tab-delimited).

Run from project root:
python training/scripts/convert_cot_to_csv.py
"""

import json
import csv
from pathlib import Path


INPUT_FILE = Path("data/raw/source/CoT_Reasoning_Quantum_Physics_And_Computing.json")
OUTPUT_FILE = Path("data/raw/source/cot_qa.csv")


def clean_text(text):
    """Clean text for CSV."""
    # Replace newlines with space
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    # Replace tabs with space
    text = text.replace('\t', ' ')
    # Remove quotes that break CSV parsing
    text = text.replace('"', "'")
    # Collapse multiple spaces
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text.strip()


def main():
    print("Converting CoT JSON to CSV...")
    
    # Load JSON
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} entries")
    
    # Write CSV with proper quoting
    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_ALL)
        writer.writerow(['question', 'answer'])
        
        for entry in data:
            question = clean_text(entry['question'])
            answer = clean_text(entry['answer'])
            writer.writerow([question, answer])
    
    print(f"Saved to {OUTPUT_FILE}")
    print(f"Done: {len(data)} Q&A pairs")
    
    # Verify
    print("\nVerifying...")
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        rows = list(reader)
        print(f"Total rows: {len(rows)}")
        print(f"Header: {rows[0]}")
        print(f"All rows have 2 columns: {all(len(r) == 2 for r in rows)}")


if __name__ == "__main__":
    main()
