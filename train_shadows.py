"""
Train shadow models and collect the attack training dataset.

For a full local run:
    python train_shadows.py --num_shadows 50 --train_size 10000

For a SLURM array task (one shadow model per task):
    python train_shadows.py --start_idx $SLURM_ARRAY_TASK_ID --end_idx $((SLURM_ARRAY_TASK_ID+1))

After all tasks finish, merge outputs:
    python train_shadows.py --merge_only --num_shadows 50 --save_dir results/shadows/
"""
import argparse
import os
import torch

from src.data_utils import load_cifar10, partition_data
from src.shadow_models import train_shadow_models, merge_shadow_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_shadows', type=int,   default=50)
    parser.add_argument('--train_size',  type=int,   default=10000)
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--lr',          type=float, default=0.001)
    parser.add_argument('--batch_size',  type=int,   default=64)
    parser.add_argument('--data_dir',    type=str,   default='./data')
    parser.add_argument('--save_dir',    type=str,   default='./results/shadows')
    parser.add_argument('--start_idx',   type=int,   default=0)
    parser.add_argument('--end_idx',     type=int,   default=None)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--merge_only',  action='store_true',
                        help='Skip training; just merge existing shadow data files')
    args = parser.parse_args()

    if args.end_idx is None:
        args.end_idx = args.num_shadows

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    merged_path = os.path.join(args.save_dir, 'attack_data.pt')

    is_partial_run = (args.start_idx != 0 or args.end_idx != args.num_shadows)

    if not args.merge_only:
        full_train, _ = load_cifar10(args.data_dir)
        _, d_shadow_pool = partition_data(full_train, seed=args.seed)

        print(f"Training shadow models {args.start_idx}–{args.end_idx-1} | "
              f"train_size={args.train_size} | epochs={args.epochs}")

        train_shadow_models(
            d_shadow_pool=d_shadow_pool,
            num_shadows=args.num_shadows,
            train_size=args.train_size,
            save_dir=args.save_dir,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
        )

    if args.merge_only or not is_partial_run:
        print(f"\nMerging shadow data -> {merged_path}")
        merge_shadow_data(args.save_dir, args.num_shadows, merged_path)
    else:
        print(f"\nPartial run complete (shadows {args.start_idx}–{args.end_idx-1}).")
        print(f"Run with --merge_only after all tasks finish to create attack_data.pt")


if __name__ == '__main__':
    main()
