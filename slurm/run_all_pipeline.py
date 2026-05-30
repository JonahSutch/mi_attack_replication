#!/usr/bin/env python3
"""
Orchestrate the entire membership inference replication pipeline using Slurm dependencies.
Run once from the repository root:
    python slurm/run_all_pipeline.py
"""
import os
import subprocess
import sys

PYTHON_EXEC = os.environ.get("PYTHON_EXEC", "/nfs/stak/users/leond/.conda/envs/trustworthy_ml/bin/python")

def run_command(cmd, shell=False):
    """Run a command, returning its stdout stripped, and crash on error."""
    # Ensure stdout/stderr decoding handles any environment quirks
    res = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error executing: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        print(f"Stdout:\n{res.stdout}")
        print(f"Stderr:\n{res.stderr}")
        sys.exit(res.returncode)
    return res.stdout.strip()

def main():
    # Make sure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Step 1: Run setup and pre-download dataset
    print("=== Step 1: Running setup and pre-downloading dataset ===")
    run_command(["bash", "slurm/setup.sh"])

    # Step 2 & 3: Submit target and shadow training jobs
    print("=== Step 2 & 3: Submitting target and shadow training jobs ===")
    print("Submitting target training jobs...")
    target_2500 = run_command(["sbatch", "--parsable", "slurm/train_target.sh", "--train_size", "2500"])
    target_5000 = run_command(["sbatch", "--parsable", "slurm/train_target.sh", "--train_size", "5000"])
    target_10000 = run_command(["sbatch", "--parsable", "slurm/train_target.sh", "--train_size", "10000"])
    target_15000 = run_command(["sbatch", "--parsable", "slurm/train_target.sh", "--train_size", "15000"])

    print("Submitting shadow model training array (100 tasks)...")
    shadow_id = run_command(["sbatch", "--parsable", "slurm/train_shadows.sh"])

    # Step 4: Schedule merge job (CPU-bound, waits for shadow array to complete)
    print("=== Step 4: Scheduling merge job ===")
    merge_cmd = [
        "sbatch", "--parsable",
        "--job-name=mi_merge",
        "--output=logs/merge_%j.out",
        "--error=logs/merge_%j.err",
        "--time=00:10:00",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--mem=4G",
        f"--dependency=afterok:{shadow_id}",
        f"--wrap=\"{PYTHON_EXEC} train_shadows.py --merge_only --num_shadows 100 --save_dir results/shadows\""
    ]
    merge_id = run_command(merge_cmd)

    # Step 5: Schedule attack training (GPU job, waits for merge job to complete)
    print("=== Step 5: Scheduling attack model training ===")
    attack_cmd = [
        "sbatch", "--parsable",
        f"--dependency=afterok:{merge_id}",
        "slurm/train_attack.sh"
    ]
    attack_id = run_command(attack_cmd)

    # Step 6: Schedule evaluation/plot generation (CPU job, waits for attack and all target jobs)
    print("=== Step 6: Scheduling evaluation and plot generation ===")
    eval_cmd = [
        "sbatch", "--parsable",
        "--job-name=mi_eval",
        "--output=logs/eval_%j.out",
        "--error=logs/eval_%j.err",
        "--time=00:15:00",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--mem=4G",
        f"--dependency=afterok:{attack_id}:{target_2500}:{target_5000}:{target_10000}:{target_15000}",
        f"--wrap=\"{PYTHON_EXEC} run_attack.py --sweep --plot\""
    ]
    eval_id = run_command(eval_cmd)

    print("\n=========================================================")
    print("Pipeline submitted successfully!")
    print(f"Target 2500 Job ID:  {target_2500}")
    print(f"Target 5000 Job ID:  {target_5000}")
    print(f"Target 10000 Job ID: {target_10000}")
    print(f"Target 15000 Job ID: {target_15000}")
    print(f"Shadow Array Job ID: {shadow_id}")
    print(f"Merge Job ID:        {merge_id}")
    print(f"Attack Job ID:       {attack_id}")
    print(f"Eval/Plot Job ID:    {eval_id}")
    print("=========================================================")
    print("Monitor everything with: squeue -u leond\n")

if __name__ == "__main__":
    main()
