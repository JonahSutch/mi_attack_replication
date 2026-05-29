#!/bin/bash
#SBATCH --job-name=mi_attack
#SBATCH --output=logs/attack_%A_%a.out
#SBATCH --error=logs/attack_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint="t4|rtx6000|rtx8000|a40|h100|h200|l40s"
#SBATCH --array=0-3
# Array index → target train_size:
#   0 → 2500   1 → 5000   2 → 10000   3 → 15000

# Run AFTER merge_shadows.sh array completes.
# Submit with:
#   sbatch slurm/train_attack.sh

TRAIN_SIZES=(2500 5000 10000 15000)
TRAIN_SIZE=${TRAIN_SIZES[$SLURM_ARRAY_TASK_ID]}

module load python/3.10
source ~/tml_env/bin/activate
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs results/attack_models

python3 train_attack.py \
    --train_size  "$TRAIN_SIZE" \
    --shadows_dir results/shadows \
    --save_dir    results/attack_models \
    --epochs      50