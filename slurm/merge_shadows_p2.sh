#!/bin/bash
#SBATCH --job-name=mi_merge_p2
#SBATCH --output=logs/merge_p2_%A_%a.out
#SBATCH --error=logs/merge_p2_%A_%a.err
#SBATCH --time=0:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --array=0-3
# Array index → target train_size:
#   0 → 2500   1 → 5000   2 → 10000   3 → 15000

# Runs after the corresponding train_shadows_p2.sh array finishes.

SIZES=(2500 5000 10000 15000)
TRAIN_SIZE=${SIZES[$SLURM_ARRAY_TASK_ID]}

module load python/3.10
source ~/tml_env/bin/activate
cd "$SLURM_SUBMIT_DIR"

python3 train_shadows.py \
    --merge_only \
    --num_shadows 50 \
    --save_dir results/shadows/size_${TRAIN_SIZE}
