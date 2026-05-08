"""
Train a target CNN on CIFAR-10 and save the checkpoint.

Example:
    python train_target.py --train_size 2500 --epochs 100 --save_path results/target_2500.pt
"""
import argparse
import os
import torch

from src.data_utils import load_cifar10, partition_data, get_target_split, make_loader
from src.target_model import TargetCNN, train_model, get_accuracy
from src.evaluate import compute_generalization_gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', type=int, default=10000)
    parser.add_argument('--epochs',     type=int, default=100)
    parser.add_argument('--lr',         type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_dir',   type=str, default='./data')
    parser.add_argument('--save_path',  type=str, default=None)
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = f'results/target_{args.train_size}.pt'

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Training target model | train_size={args.train_size} | epochs={args.epochs}")

    full_train, full_test = load_cifar10(args.data_dir)
    d_target_pool, _ = partition_data(full_train, seed=args.seed)
    target_train, target_nonmember = get_target_split(d_target_pool, args.train_size)

    train_loader  = make_loader(target_train,     batch_size=args.batch_size, shuffle=True)
    nonmem_loader = make_loader(target_nonmember, batch_size=args.batch_size, shuffle=False)
    test_loader   = make_loader(full_test,        batch_size=args.batch_size, shuffle=False)

    model = TargetCNN()
    train_model(model, train_loader, epochs=args.epochs, lr=args.lr, device=device)

    gap, train_acc, test_acc = compute_generalization_gap(model, train_loader, test_loader, device)
    print(f"\nTrain acc:  {train_acc:.4f}")
    print(f"Test acc:   {test_acc:.4f}")
    print(f"Gen. gap:   {gap:.4f}")

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    torch.save({
        'state_dict':  model.state_dict(),
        'train_size':  args.train_size,
        'train_acc':   train_acc,
        'test_acc':    test_acc,
        'gap':         gap,
        'seed':        args.seed,
    }, args.save_path)
    print(f"Saved to {args.save_path}")


if __name__ == '__main__':
    main()
