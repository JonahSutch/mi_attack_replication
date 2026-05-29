#!/bin/bash
#SBATCH --job-name=mi_run_attack
#SBATCH --output=logs/run_attack_%j.out
#SBATCH --error=logs/run_attack_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint="t4|rtx6000|rtx8000|a40|h100|h200|l40s"
# Run AFTER train_attack.sh array completes.
# Submit with:
#   sbatch slurm/run_attack.sh

module load python/3.10
source ~/tml_env/bin/activate
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs results/figures

pip install -r requirements.txt --quiet

python3 run_attack.py \
    --attack_models_root results/attack_models \
    --results_dir        results \
    --data_dir           data \
    --plot