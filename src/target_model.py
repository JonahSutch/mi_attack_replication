import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class TargetCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.Tanh(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def train_model(model, train_loader, epochs=100, lr=0.001, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        correct = total = 0
        for x, y in train_loader:
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
            print(f"  Epoch {epoch:3d}/{epochs}  train_acc={correct/total:.4f}")


def get_accuracy(model, loader, device='cpu'):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def get_confidence_vectors(model, loader, device='cpu'):
    """Returns (conf_tensor, label_tensor). conf_tensor is softmax output, shape (N, 10)."""
    model.eval()
    all_conf = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            conf = torch.softmax(logits, dim=1).cpu()
            all_conf.append(conf)
            all_labels.append(y)
    return torch.cat(all_conf), torch.cat(all_labels)
