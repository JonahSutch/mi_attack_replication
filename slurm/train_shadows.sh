#!/bin/bash
#SBATCH --job-name=mi_shadows
#SBATCH --output=logs/shadow_%A_%a.out
#SBATCH --error=logs/shadow_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=0-49        # one task per shadow model (0-indexed)

# Submit with:
#   sbatch slurm/train_shadows.sh
#
# After all tasks complete, merge results:
#   python train_shadows.py --merge_only --num_shadows 50

module load cuda
source activate mi_env  # adjust to your conda env name

cd "$(dirname "$0")/.."
mkdir -p logs results/shadows

TRAIN_SIZE=${TRAIN_SIZE:-10000}

python train_shadows.py \
    --num_shadows 50 \
    --train_size "$TRAIN_SIZE" \
    --epochs 100 \
    --save_dir results/shadows \
    --start_idx "$SLURM_ARRAY_TASK_ID" \
    --end_idx $((SLURM_ARRAY_TASK_ID + 1))
