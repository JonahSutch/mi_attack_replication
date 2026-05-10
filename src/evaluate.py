import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from .target_model import get_accuracy, get_confidence_vectors


def evaluate_attack(attack_models, target_model, member_loader, nonmember_loader, device='cpu'):
    """
    Query target model for confidence vectors, run attack model predictions.
    Returns dict: {accuracy, precision, recall, f1}
    """
    conf_mem,    labels_mem    = get_confidence_vectors(target_model, member_loader,    device)
    conf_nonmem, labels_nonmem = get_confidence_vectors(target_model, nonmember_loader, device)

    preds_mem    = _batch_predict(attack_models, conf_mem,    labels_mem,    device)
    preds_nonmem = _batch_predict(attack_models, conf_nonmem, labels_nonmem, device)

    y_true = torch.cat([torch.ones(len(preds_mem), dtype=torch.long),
                        torch.zeros(len(preds_nonmem), dtype=torch.long)]).numpy()
    y_pred = torch.cat([preds_mem, preds_nonmem]).numpy()

    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
    }


def _batch_predict(attack_models, conf_tensor, label_tensor, device):
    """Run all 10 per-class attack models in one forward pass each, then route by true label."""
    n = len(conf_tensor)
    preds = torch.zeros(n, dtype=torch.long)
    conf_tensor = conf_tensor.to(device)

    for c, model in enumerate(attack_models):
        mask = (label_tensor == c)
        if mask.sum() == 0:
            continue
        model.eval()
        with torch.no_grad():
            logits = model(conf_tensor[mask])
            preds[mask] = logits.argmax(dim=1).cpu()

    return preds


def compute_generalization_gap(target_model, train_loader, test_loader, device='cpu'):
    train_acc = get_accuracy(target_model, train_loader, device)
    test_acc  = get_accuracy(target_model, test_loader,  device)
    return train_acc - test_acc, train_acc, test_acc


def plot_accuracy_vs_gap(results, save_path):
    """
    results: list of dicts, each with keys:
        'train_size', 'gap', 'attack_accuracy', 'precision', 'recall'
    Saves a scatter/line plot to save_path.
    """
    results_sorted = sorted(results, key=lambda r: r['gap'])
    gaps       = [r['gap']            for r in results_sorted]
    accs       = [r['attack_accuracy'] for r in results_sorted]
    precisions = [r['precision']       for r in results_sorted]
    recalls    = [r['recall']          for r in results_sorted]
    sizes      = [r['train_size']      for r in results_sorted]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(gaps, accs,       marker='o', label='Attack Accuracy')
    ax.plot(gaps, precisions, marker='s', label='Precision')
    ax.plot(gaps, recalls,    marker='^', label='Recall')
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Random baseline')

    for gap, acc, size in zip(gaps, accs, sizes):
        ax.annotate(f'n={size}', (gap, acc), textcoords='offset points', xytext=(5, 5), fontsize=8)

    ax.set_xlabel('Generalization Gap (train_acc - test_acc)')
    ax.set_ylabel('Score')
    ax.set_title('MI Attack Performance vs. Generalization Gap (CIFAR-10)')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def plot_attack_vs_baseline(results, save_path):
    """Bar chart: attack accuracy vs. 0.5 random baseline for each training size."""
    results_sorted = sorted(results, key=lambda r: r['train_size'])
    sizes  = [str(r['train_size']) for r in results_sorted]
    accs   = [r['attack_accuracy'] for r in results_sorted]
    precs  = [r['precision']       for r in results_sorted]

    x = range(len(sizes))
    width = 0.3

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar([i - width/2 for i in x], accs,  width, label='Attack Accuracy', color='steelblue')
    bars2 = ax.bar([i + width/2 for i in x], precs, width, label='Precision',       color='darkorange')
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Random baseline (0.50)')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f'n={s}' for s in sizes])
    ax.set_ylabel('Score')
    ax.set_title('MI Attack Accuracy vs. Random Baseline (CIFAR-10)')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def plot_generalization_gaps(results, save_path):
    """Grouped bar chart: train vs. test accuracy for each training size."""
    results_sorted = sorted(results, key=lambda r: r['train_size'])
    sizes      = [str(r['train_size']) for r in results_sorted]
    train_accs = [r['train_acc']       for r in results_sorted]
    test_accs  = [r['test_acc']        for r in results_sorted]

    x = range(len(sizes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([i - width/2 for i in x], train_accs, width, label='Train Accuracy', color='steelblue')
    ax.bar([i + width/2 for i in x], test_accs,  width, label='Test Accuracy',  color='darkorange')

    for i, (tr, te) in enumerate(zip(train_accs, test_accs)):
        gap = tr - te
        ax.text(i, max(tr, te) + 0.02, f'gap={gap:.2f}', ha='center', fontsize=8, color='dimgray')

    ax.set_xticks(list(x))
    ax.set_xticklabels([f'n={s}' for s in sizes])
    ax.set_ylabel('Accuracy')
    ax.set_title('Target Model Generalization Gap by Training Size (CIFAR-10)')
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def print_results_table(results):
    print(f"\n{'Train Size':>12}  {'Gap':>8}  {'Atk Acc':>8}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("-" * 65)
    for r in sorted(results, key=lambda x: x['train_size']):
        print(f"{r['train_size']:>12}  {r['gap']:>8.4f}  {r['attack_accuracy']:>8.4f}  "
              f"{r['precision']:>10.4f}  {r['recall']:>8.4f}  {r['f1']:>8.4f}")
