"""
Evaluate the membership inference attack against a trained target model.

Single evaluation:
    python run_attack.py --target_path results/target_10000.pt \
                         --attack_models_dir results/attack_models/ \
                         --train_size 10000

Full sweep (all training sizes → plot):
    python run_attack.py --sweep --attack_models_dir results/attack_models/
"""
import argparse
import os
import torch

from src.data_utils import load_cifar10, partition_data, get_target_split, make_loader
from src.target_model import TargetCNN
from src.attack_model import load_attack_models
from src.evaluate import (evaluate_attack, compute_generalization_gap,
                           plot_accuracy_vs_gap, print_results_table)


def run_single(target_path, attack_models_dir, data_dir, train_size, seed, device, batch_size):
    ckpt = torch.load(target_path, map_location=device)
    model = TargetCNN()
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()

    full_train, full_test = load_cifar10(data_dir)
    d_target_pool, _ = partition_data(full_train, seed=seed)
    target_train, target_nonmember = get_target_split(d_target_pool, train_size)

    train_loader  = make_loader(target_train,     batch_size=batch_size, shuffle=False)
    nonmem_loader = make_loader(target_nonmember, batch_size=batch_size, shuffle=False)
    test_loader   = make_loader(full_test,        batch_size=batch_size, shuffle=False)

    gap, train_acc, test_acc = compute_generalization_gap(model, train_loader, test_loader, device)

    attack_models = load_attack_models(attack_models_dir, device=device)
    metrics = evaluate_attack(attack_models, model, train_loader, nonmem_loader, device)

    return {
        'train_size':      train_size,
        'gap':             gap,
        'train_acc':       train_acc,
        'test_acc':        test_acc,
        'attack_accuracy': metrics['accuracy'],
        'precision':       metrics['precision'],
        'recall':          metrics['recall'],
        'f1':              metrics['f1'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_path',       type=str, default=None)
    parser.add_argument('--attack_models_dir', type=str, default='./results/attack_models')
    parser.add_argument('--data_dir',          type=str, default='./data')
    parser.add_argument('--train_size',        type=int, default=10000)
    parser.add_argument('--seed',              type=int, default=42)
    parser.add_argument('--batch_size',        type=int, default=256)
    parser.add_argument('--sweep',             action='store_true',
                        help='Run over all train sizes: 2500, 5000, 10000, 15000')
    parser.add_argument('--plot',              action='store_true',
                        help='Save accuracy-vs-gap plot (requires --sweep)')
    parser.add_argument('--results_dir',       type=str, default='./results')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    if args.sweep:
        train_sizes = [2500, 5000, 10000, 15000]
        results = []
        for ts in train_sizes:
            path = os.path.join(args.results_dir, f'target_{ts}.pt')
            if not os.path.exists(path):
                print(f"Missing {path}, skipping.")
                continue
            print(f"\n=== train_size={ts} ===")
            r = run_single(path, args.attack_models_dir, args.data_dir,
                           ts, args.seed, device, args.batch_size)
            results.append(r)
            print(f"  gap={r['gap']:.4f}  attack_acc={r['attack_accuracy']:.4f}  "
                  f"prec={r['precision']:.4f}  recall={r['recall']:.4f}")

        print_results_table(results)

        if args.plot and results:
            plot_path = os.path.join(args.results_dir, 'figures', 'attack_accuracy_vs_gap.png')
            plot_accuracy_vs_gap(results, plot_path)
    else:
        if args.target_path is None:
            args.target_path = os.path.join(args.results_dir, f'target_{args.train_size}.pt')
        r = run_single(args.target_path, args.attack_models_dir, args.data_dir,
                       args.train_size, args.seed, device, args.batch_size)
        print(f"\ntrain_size={r['train_size']}  gap={r['gap']:.4f}")
        print(f"Attack accuracy:  {r['attack_accuracy']:.4f}")
        print(f"Precision:        {r['precision']:.4f}")
        print(f"Recall:           {r['recall']:.4f}")
        print(f"F1:               {r['f1']:.4f}")


if __name__ == '__main__':
    main()
