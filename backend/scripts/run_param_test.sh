#!/bin/bash
#SBATCH --job-name=param_test
#SBATCH --output=param_test_%j.out
#SBATCH --error=param_test_%j.err
#SBATCH --partition=dgxh
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G

echo "Starting parameter battery test"
echo "Date: $(date)"
echo "Node: $(hostname)"

# Activate environment
source ~/hpc-share/quantum-llm/venv/bin/activate

# Navigate to scripts directory
cd ~/hpc-share/quantum-llm/scripts

# Check files
echo ""
echo "Checking files..."
ls -la cached_contexts.json 2>/dev/null || echo "WARNING: cached_contexts.json not found!"
ls -la ../model/final_model.pt 2>/dev/null || echo "WARNING: model not found!"
ls -la ../../tokenizer/tokenizer.json 2>/dev/null || echo "WARNING: tokenizer not found!"

# Run test
echo ""
echo "Running parameter battery test..."
PYTHONUNBUFFERED=1 python param_battery_hpc.py

echo ""
echo "Completed: $(date)"
