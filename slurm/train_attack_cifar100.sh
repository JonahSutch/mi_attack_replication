#!/bin/bash
#SBATCH --job-name=mi_attack_c100
#SBATCH --output=logs/attack_c100_%j.out
#SBATCH --error=logs/attack_c100_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint="t4|rtx6000|rtx8000|a40|h100|h200|l40s"

# Run after train_shadows.sh array + merge step complete.
# Submit with:
#   sbatch slurm/train_attack.sh

module load python/3.10
source ~/tml_env/bin/activate

cd "$HOME/mi_attack_replication"
mkdir -p logs results/attack_models_cifar100

python3 train_attack.py \
    --attack_data results/shadows_cifar100/attack_data.pt \
    --save_dir results/attack_models_cifar100 \
    --epochs 50
