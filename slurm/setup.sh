#!/bin/bash
# Run once on HPC before submitting any jobs.
# On HPC: bash slurm/setup.sh

set -e

# module load python/3.10
PYTHON_EXEC="${PYTHON_EXEC:-/nfs/stak/users/leond/.conda/envs/trustworthy_ml/bin/python}"

# Install missing packages into existing environment
# source ~/tml_env/bin/activate
"$PYTHON_EXEC" -m pip install scikit-learn tqdm --quiet

# Pre-download CIFAR-10 to avoid race conditions when 100 array tasks start at once
cd "$(dirname "$0")/.."
mkdir -p data results/shadows results/attack_models results/figures logs

"$PYTHON_EXEC" -c "
import torchvision
torchvision.datasets.CIFAR10(root='data', train=True,  download=True)
torchvision.datasets.CIFAR10(root='data', train=False, download=True)
print('CIFAR-10 downloaded.')
"

echo "Setup complete."
