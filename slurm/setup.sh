#!/bin/bash
# Run once on HPC before submitting any jobs.
# On HPC: bash slurm/setup.sh

set -e

module load python/3.10

# Install missing packages into existing venv
source ~/tml_env/bin/activate
pip install scikit-learn tqdm --quiet

# Pre-download CIFAR-10 to avoid race conditions when 50 array tasks start at once
cd "$(dirname "$0")/.."
mkdir -p data results/shadows results/attack_models results/figures logs

python3 -c "
import torchvision
torchvision.datasets.CIFAR10(root='data', train=True,  download=True)
torchvision.datasets.CIFAR10(root='data', train=False, download=True)
print('CIFAR-10 downloaded.')
"

echo "Setup complete."
