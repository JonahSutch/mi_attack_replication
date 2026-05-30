#!/bin/bash
# Submit all independent training jobs at once.
# Run from ~/mi_attack_replication on HPC:
#   bash slurm/submit_all.sh
#
# After all jobs finish, run the merge + attack + sweep manually:
#   /nfs/stak/users/leond/.conda/envs/trustworthy_ml/bin/python train_shadows.py --merge_only --num_shadows 100 --save_dir results/shadows
#   sbatch slurm/train_attack.sh
#   (after attack finishes) /nfs/stak/users/leond/.conda/envs/trustworthy_ml/bin/python run_attack.py --sweep --plot

set -e
cd "$(dirname "$0")/.."
mkdir -p logs

PYTHON_EXEC="${PYTHON_EXEC:-/nfs/stak/users/leond/.conda/envs/trustworthy_ml/bin/python}"

echo "Submitting target model jobs..."
sbatch slurm/train_target.sh --train_size 2500
sbatch slurm/train_target.sh --train_size 5000
sbatch slurm/train_target.sh --train_size 10000
sbatch slurm/train_target.sh --train_size 15000

echo "Submitting shadow model array (100 tasks)..."
sbatch slurm/train_shadows.sh

echo ""
echo "All jobs submitted. Monitor with: squeue -u leond"
echo ""
echo "When all finish, run:"
echo "  $PYTHON_EXEC train_shadows.py --merge_only --num_shadows 100 --save_dir results/shadows"
echo "  sbatch slurm/train_attack.sh"
