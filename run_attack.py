"""
Evaluate the P2 per-size attack models against each target model and
synthesise results across all train_size configurations.

Uses the same run_single() logic as run_attack.py, but points each
train_size at its own dedicated attack model directory instead of a
single shared one.

Example:
    python run_attack.py
    python run_attack.py --plot
    python run_attack.py --train_sizes 2500 5000  # subset

Output: prints a results table and (optionally) saves figures to
results/figures/ using the same plot helpers as run_attack.py.
"""

import argparse
import os

import torch

from run_attack import run_single
from src.evaluate import (plot_accuracy_vs_gap, plot_attack_vs_baseline,
                          plot_generalization_gaps, print_results_table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_sizes', type=int, nargs='+',
                        default=[2500, 5000, 10000, 15000],
                        help='Target train sizes to evaluate.')
    parser.add_argument('--attack_models_root', type=str,
                        default='./results/attack_models',
                        help='Root dir written by train_attack.py; '
                             'expects size_<N>/ subdirs.')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--data_dir',    type=str, default='./data')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--batch_size',  type=int, default=256)
    parser.add_argument('--plot', action='store_true',
                        help='Save figures to results/figures/.')
    args = parser.parse_args()

    device = ('cuda'  if torch.cuda.is_available() else
              'mps'   if torch.backends.mps.is_available() else
              'cpu')
    print(f"Device: {device}")

    results = []
    for ts in args.train_sizes:
        target_path      = os.path.join(args.results_dir, f'target_{ts}.pt')
        attack_models_dir = os.path.join(args.attack_models_root, f'size_{ts}')

        if not os.path.exists(target_path):
            print(f"Missing {target_path}, skipping.")
            continue
        if not os.path.isdir(attack_models_dir):
            print(f"Missing attack models dir {attack_models_dir}, skipping.")
            continue

        print(f"\n=== train_size={ts} | attack_dir=.../{os.path.basename(attack_models_dir)} ===")
        r = run_single(target_path, attack_models_dir, args.data_dir,
                       ts, args.seed, device, args.batch_size)
        results.append(r)
        print(f"  gap={r['gap']:.4f}  attack_acc={r['attack_accuracy']:.4f}  "
              f"prec={r['precision']:.4f}  recall={r['recall']:.4f}")

    if not results:
        print("\nNo results collected — check paths and run training first.")
        return

    print_results_table(results)

    if args.plot:
        fig_dir = os.path.join(args.results_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        plot_accuracy_vs_gap(results,
            os.path.join(fig_dir, 'attack_accuracy_vs_gap.png'))
        plot_attack_vs_baseline(results,
            os.path.join(fig_dir, 'attack_vs_baseline.png'))
        plot_generalization_gaps(results,
            os.path.join(fig_dir, 'generalization_gaps.png'))
        print(f"\nFigures saved to {fig_dir}/")


if __name__ == '__main__':
    main()
