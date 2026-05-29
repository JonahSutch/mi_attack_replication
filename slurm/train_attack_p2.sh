#!/bin/bash
#SBATCH --job-name=mi_attack_p2
#SBATCH --output=logs/attack_p2_%A_%a.out
#SBATCH --error=logs/attack_p2_%A_%a.err
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
#
# Called by submit_p2.sh after merge_shadows_p2.sh completes.
# Can also be submitted manually:
#   sbatch slurm/train_attack_p2.sh

TRAIN_SIZES=(2500 5000 10000 15000)
TRAIN_SIZE=${TRAIN_SIZES[$SLURM_ARRAY_TASK_ID]}

module load python/3.10
source ~/tml_env/bin/activate
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs results/attack_models_p2

python3 train_attack_p2.py \
    --train_size  "$TRAIN_SIZE" \
    --shadows_dir results/shadows \
    --save_dir    results/attack_models_p2 \
    --epochs      50