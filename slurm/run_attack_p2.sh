#!/bin/bash
#SBATCH --job-name=mi_run_attack_p2
#SBATCH --output=logs/run_attack_p2_%j.out
#SBATCH --error=logs/run_attack_p2_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint="t4|rtx6000|rtx8000|a40|h100|h200|l40s"
# Run AFTER train_attack_p2.sh array completes.
# Submit with:
#   sbatch slurm/run_attack_p2.sh

module load python/3.10
source ~/tml_env/bin/activate
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs results/figures

pip install -r requirements.txt --quiet

python3 run_attack_p2.py \
    --attack_models_root results/attack_models_p2 \
    --results_dir        results \
    --data_dir           data \
    --plot