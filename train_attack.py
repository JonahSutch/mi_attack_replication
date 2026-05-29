"""
Train 10 per-class attack MLP models for a single target train_size (P2 tests).

Each size uses its own shadow-model attack dataset so the attack models are
trained on confidences from shadows that match the target's training regime.

Example:
    python train_attack.py --train_size 2500
    python train_attack.py --train_size 10000

The SLURM array job (slurm/train_attack.sh) calls this once per size.

Output layout:
    results/attack_models/
        size_2500/    class_0.pt ... class_9.pt
        size_5000/    ...
        size_10000/   ...
        size_15000/   ...
"""
import argparse
import os
import torch
from src.shadow_models import load_attack_data
from src.attack_model import train_attack_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size',  type=int, required=True,
                        help='Target model train_size (2500 / 5000 / 10000 / 15000).')
    parser.add_argument('--shadows_dir', type=str, default='./results/shadows',
                        help='Root shadows dir; attack data loaded from '
                             '<shadows_dir>/size_<train_size>/attack_data.pt')
    parser.add_argument('--save_dir',    type=str, default='./results/attack_models')
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--lr',          type=float, default=0.001)
    parser.add_argument('--batch_size',  type=int,   default=256)
    args = parser.parse_args()

    device = ('cuda'  if torch.cuda.is_available() else
              'mps'   if torch.backends.mps.is_available() else
              'cpu')

    attack_data_path = os.path.join(
        args.shadows_dir, f'size_{args.train_size}', 'attack_data.pt')
    size_dir = os.path.join(args.save_dir, f'size_{args.train_size}')
    os.makedirs(size_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"train_size={args.train_size} | attack_data={attack_data_path}")

    if not os.path.exists(attack_data_path):
        raise FileNotFoundError(
            f"{attack_data_path} not found.\n"
            f"Run train_shadows.sh + merge_shadows.sh for size={args.train_size} first."
        )

    attack_data = load_attack_data(attack_data_path)
    n = len(attack_data['conf'])
    n_in = attack_data['in_out'].sum().item()
    print(f"Total examples: {n}  (in={int(n_in)}, out={n - int(n_in)})")
    print(f"Training 10 per-class attack models | epochs={args.epochs}")

    train_attack_models(
        attack_data=attack_data,
        save_dir=size_dir,
        num_classes=10,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
    )

    print(f"\nAttack models (train_size={args.train_size}) saved to {size_dir}")


if __name__ == '__main__':
    main()
