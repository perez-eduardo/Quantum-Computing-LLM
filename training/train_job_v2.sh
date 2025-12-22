#!/bin/bash
#SBATCH --job-name=quantum-llm-v2
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=dgxh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Quantum Computing LLM Training Job v2
# Changes from v1:
#   - Epochs: 3 -> 10 (model wasn't converged)
#   - Min LR: 3e-5 -> 1e-5 (allow longer training at low LR)
#   - Warmup ratio: 0.1 -> 0.05 (faster warmup)

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

# Navigate to project directory
cd ~/hpc-share/quantum-llm

# Activate virtual environment
source venv/bin/activate

# Load required modules
module load python/3.11

# Create logs directory
mkdir -p logs

# Print environment info
echo ""
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""

# Navigate to scripts directory
cd scripts

# Run training with updated hyperparameters
python train.py \
    --qa_path ../data/combined_qa_final.csv \
    --book_path ../data/combined_books.txt \
    --tokenizer_path ../tokenizer.json \
    --output_dir ../model \
    --dim 64 \
    --n_layers 4 \
    --n_heads 4 \
    --ff_dim 256 \
    --max_length 512 \
    --epochs 10 \
    --batch_size 64 \
    --max_lr 3e-4 \
    --min_lr 1e-5 \
    --warmup_ratio 0.05 \
    --log_interval 100

echo ""
echo "========================================"
echo "End time: $(date)"
echo "========================================"
