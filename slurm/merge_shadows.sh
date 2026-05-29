#!/bin/bash
#SBATCH --job-name=mi_shadows
#SBATCH --output=logs/merge_shadows_%j.out
#SBATCH --error=logs/merge_shadows_%j.err
#SBATCH --time=0:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=gpu

# Submit with:
#   sbatch slurm/merge_shadows.sh

module load python/3.10
source ~/tml_env/bin/activate

cd "$SLURM_SUBMIT_DIR"

python3 train_shadows.py --merge_only --num_shadows 50 --save_dir results/shadows