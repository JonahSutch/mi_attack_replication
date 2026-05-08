import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class AttackMLP(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_attack_models(attack_data, save_dir, num_classes=10, epochs=50,
                        lr=0.001, batch_size=256, device='cpu'):
    """
    Train one AttackMLP per class. Returns list of 10 trained models.
    attack_data: dict with keys 'conf' (N,10), 'true_label' (N,), 'in_out' (N,)
    """
    os.makedirs(save_dir, exist_ok=True)
    conf       = attack_data['conf']
    true_label = attack_data['true_label']
    in_out     = attack_data['in_out']

    criterion = nn.CrossEntropyLoss()
    models = []

    for c in range(num_classes):
        mask = (true_label == c)
        conf_c   = conf[mask]
        in_out_c = in_out[mask]

        print(f"\nClass {c}: {mask.sum().item()} examples "
              f"({in_out_c.sum().item()} in, {(~in_out_c.bool()).sum().item()} out)")

        dataset = TensorDataset(conf_c, in_out_c)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = AttackMLP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            model.train()
            correct = total = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Class {c}  Epoch {epoch:3d}/{epochs}  train_acc={correct/total:.4f}")

        ckpt_path = os.path.join(save_dir, f"attack_model_class_{c}.pt")
        torch.save(model.state_dict(), ckpt_path)
        models.append(model)

    return models


def load_attack_models(save_dir, num_classes=10, device='cpu'):
    models = []
    for c in range(num_classes):
        model = AttackMLP()
        state = torch.load(os.path.join(save_dir, f"attack_model_class_{c}.pt"),
                           map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()
        models.append(model)
    return models


def predict_membership(attack_models, conf_vec, true_label, device='cpu'):
    """
    conf_vec: tensor of shape (10,) — softmax output from target model
    true_label: int — the ground-truth class of the query point
    Returns: 1 (member) or 0 (non-member)
    """
    model = attack_models[true_label]
    model.eval()
    with torch.no_grad():
        x = conf_vec.unsqueeze(0).to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).item()
    return pred
