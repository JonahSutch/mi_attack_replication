"""
Train 10 per-class attack MLP models from the saved attack dataset.

Example:
    python train_attack.py --attack_data results/shadows/attack_data.pt
"""
import argparse
import torch

from src.shadow_models import load_attack_data
from src.attack_model import train_attack_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_data', type=str, default='./results/shadows/attack_data.pt')
    parser.add_argument('--save_dir',    type=str, default='./results/attack_models')
    parser.add_argument('--epochs',      type=int, default=50)
    parser.add_argument('--lr',          type=float, default=0.001)
    parser.add_argument('--batch_size',  type=int, default=256)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Loading attack data from {args.attack_data}")

    attack_data = load_attack_data(args.attack_data)
    n = len(attack_data['conf'])
    n_in  = attack_data['in_out'].sum().item()
    print(f"Total examples: {n}  (in={n_in}, out={n - n_in})")

    print(f"Training 10 per-class attack models | epochs={args.epochs}")
    train_attack_models(
        attack_data=attack_data,
        save_dir=args.save_dir,
        num_classes=10,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
    )
    print(f"\nAttack models saved to {args.save_dir}")


if __name__ == '__main__':
    main()
