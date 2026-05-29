#!/bin/bash
#SBATCH --job-name=mi_shadows_p2
#SBATCH --output=logs/shadow_p2_%A_%a.out
#SBATCH --error=logs/shadow_p2_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint="t4|rtx6000|rtx8000|a40|h100|h200|l40s"
#SBATCH --array=0-149

SIZES=(2500 5000 10000 15000)
size_idx=$(( $SLURM_ARRAY_TASK_ID / 50 ))
local_idx=$(( $SLURM_ARRAY_TASK_ID % 50 ))
TRAIN_SIZE=${SIZES[$size_idx]}

module load python/3.10
source ~/tml_env/bin/activate
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/shadows/size_${TRAIN_SIZE}

python3 train_shadows.py \
    --num_shadows 50 \
    --train_size  "$TRAIN_SIZE" \
    --epochs      100 \
    --save_dir    results/shadows/size_${TRAIN_SIZE} \
    --start_idx   "$local_idx" \
    --end_idx     $(($local_idx + 1))
