import os
import torch
from torch.utils.data import TensorDataset

from .data_utils import get_shadow_split, make_loader
from .target_model import TargetCNN, train_model, get_confidence_vectors


def train_shadow_models(d_shadow_pool, num_shadows, train_size, save_dir,
                        epochs=100, lr=0.001, batch_size=64, device='cpu',
                        start_idx=0, end_idx=None):
    """
    Train shadow models from start_idx to end_idx (exclusive).
    Each model's confidence vectors are saved to save_dir/shadow_{i}_data.pt
    so SLURM array tasks can run independently and be merged later.
    """
    if end_idx is None:
        end_idx = num_shadows
    os.makedirs(save_dir, exist_ok=True)

    for i in range(start_idx, end_idx):
        out_path = os.path.join(save_dir, f"shadow_{i}_data.pt")
        if os.path.exists(out_path):
            print(f"Shadow {i}: already exists, skipping.")
            continue

        print(f"\n--- Shadow model {i+1}/{num_shadows} ---")
        shadow_train, shadow_test = get_shadow_split(d_shadow_pool, train_size, seed=i)

        train_loader = make_loader(shadow_train, batch_size=batch_size, shuffle=True)
        test_loader  = make_loader(shadow_test,  batch_size=batch_size, shuffle=False)

        model = TargetCNN().to(device)
        train_model(model, train_loader, epochs=epochs, lr=lr, device=device)

        conf_in,  labels_in  = get_confidence_vectors(model, train_loader, device)
        conf_out, labels_out = get_confidence_vectors(model, test_loader,  device)

        # in_out_label: 1 = member, 0 = non-member
        in_out_in  = torch.ones(len(conf_in),  dtype=torch.long)
        in_out_out = torch.zeros(len(conf_out), dtype=torch.long)

        conf   = torch.cat([conf_in,   conf_out])
        labels = torch.cat([labels_in, labels_out])
        in_out = torch.cat([in_out_in, in_out_out])

        torch.save({'conf': conf, 'true_label': labels, 'in_out': in_out}, out_path)
        print(f"  Saved {len(conf)} examples to {out_path}")

        # Optionally save model checkpoint
        ckpt_path = os.path.join(save_dir, f"shadow_{i}.pt")
        torch.save(model.state_dict(), ckpt_path)


def merge_shadow_data(save_dir, num_shadows, output_path):
    """
    Merge per-shadow .pt files into a single attack_data.pt.
    Call this after all SLURM array tasks complete.
    """
    all_conf, all_labels, all_in_out = [], [], []
    for i in range(num_shadows):
        path = os.path.join(save_dir, f"shadow_{i}_data.pt")
        if not os.path.exists(path):
            print(f"Warning: missing shadow_{i}_data.pt, skipping.")
            continue
        d = torch.load(path)
        all_conf.append(d['conf'])
        all_labels.append(d['true_label'])
        all_in_out.append(d['in_out'])

    merged = {
        'conf':       torch.cat(all_conf),
        'true_label': torch.cat(all_labels),
        'in_out':     torch.cat(all_in_out),
    }
    torch.save(merged, output_path)
    print(f"Merged {len(merged['conf'])} total examples -> {output_path}")
    return merged


def load_attack_data(path):
    return torch.load(path)
