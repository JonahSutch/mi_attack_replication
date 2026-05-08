#!/bin/bash
#SBATCH --job-name=mi_target
#SBATCH --output=logs/target_%j.out
#SBATCH --error=logs/target_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Usage:
#   sbatch slurm/train_target.sh --train_size 2500
#   sbatch slurm/train_target.sh --train_size 5000
#   sbatch slurm/train_target.sh --train_size 10000
#   sbatch slurm/train_target.sh --train_size 15000

module load cuda
source activate mi_env  # adjust to your conda env name

cd "$(dirname "$0")/.."
mkdir -p logs results

python train_target.py --epochs 100 "$@"
