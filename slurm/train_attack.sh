#!/bin/bash
#SBATCH --job-name=mi_attack
#SBATCH --output=logs/attack_%j.out
#SBATCH --error=logs/attack_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Run after train_shadows.sh array + merge step complete.
# Submit with:
#   sbatch slurm/train_attack.sh

module load cuda
source activate mi_env  # adjust to your conda env name

cd "$(dirname "$0")/.."
mkdir -p logs results/attack_models

python train_attack.py \
    --attack_data results/shadows/attack_data.pt \
    --save_dir results/attack_models \
    --epochs 50
