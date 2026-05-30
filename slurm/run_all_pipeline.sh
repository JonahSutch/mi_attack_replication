#!/bin/bash
# Coordinate the entire pipeline execution on the HPC cluster using Slurm job dependencies.
# Run once from the repository root:
#   bash slurm/run_all_pipeline.sh

set -e
cd "$(dirname "$0")/.."

PYTHON_EXEC="${PYTHON_EXEC:-/nfs/stak/users/leond/.conda/envs/trustworthy_ml/bin/python}"

echo "=== Step 1: Running setup and pre-downloading dataset ==="
bash slurm/setup.sh

echo "=== Step 2 & 3: Submitting target and shadow training jobs ==="
echo "Submitting target training jobs..."
TARGET_2500_ID=$(sbatch --parsable slurm/train_target.sh --train_size 2500)
TARGET_5000_ID=$(sbatch --parsable slurm/train_target.sh --train_size 5000)
TARGET_10000_ID=$(sbatch --parsable slurm/train_target.sh --train_size 10000)
TARGET_15000_ID=$(sbatch --parsable slurm/train_target.sh --train_size 15000)

echo "Submitting shadow model training array (100 tasks)..."
SHADOW_ID=$(sbatch --parsable slurm/train_shadows.sh)

echo "=== Step 4: Scheduling merge job ==="
# Merge shadow model outputs after the shadow array finishes.
# Since it is a CPU-only and extremely short task, we run it as a short CPU job.
MERGE_ID=$(sbatch --parsable \
    --job-name=mi_merge \
    --output=logs/merge_%j.out \
    --error=logs/merge_%j.err \
    --time=00:10:00 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=4G \
    --dependency=afterok:$SHADOW_ID \
    --wrap="$PYTHON_EXEC train_shadows.py --merge_only --num_shadows 100 --save_dir results/shadows")

echo "=== Step 5: Scheduling attack model training ==="
# Train attack models after merging complete
ATTACK_ID=$(sbatch --parsable \
    --dependency=afterok:$MERGE_ID \
    slurm/train_attack.sh)

echo "=== Step 6: Scheduling evaluation and plot generation ==="
# Run final evaluation once the attack models and all target models are trained
EVAL_ID=$(sbatch --parsable \
    --job-name=mi_eval \
    --output=logs/eval_%j.out \
    --error=logs/eval_%j.err \
    --time=00:15:00 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=4G \
    --dependency=afterok:$ATTACK_ID:$TARGET_2500_ID:$TARGET_5000_ID:$TARGET_10000_ID:$TARGET_15000_ID \
    --wrap="$PYTHON_EXEC run_attack.py --sweep --plot")

echo ""
echo "========================================================="
echo "Pipeline submitted successfully!"
echo "Target 2500 Job ID:  $TARGET_2500_ID"
echo "Target 5000 Job ID:  $TARGET_5000_ID"
echo "Target 10000 Job ID: $TARGET_10000_ID"
echo "Target 15000 Job ID: $TARGET_15000_ID"
echo "Shadow Array Job ID: $SHADOW_ID"
echo "Merge Job ID:        $MERGE_ID"
echo "Attack Job ID:       $ATTACK_ID"
echo "Eval/Plot Job ID:    $EVAL_ID"
echo "========================================================="
echo "Monitor everything with: squeue -u leond"
echo ""
