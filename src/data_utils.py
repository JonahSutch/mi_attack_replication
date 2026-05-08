import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def load_cifar10(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=transform)
    test_set  = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    return train_set, test_set


def partition_data(full_train, seed=42):
    """Split 50k CIFAR-10 train set into D_target_pool and D_shadow_pool (25k each)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(full_train), generator=g).tolist()
    d_target_pool = Subset(full_train, idx[:25000])
    d_shadow_pool  = Subset(full_train, idx[25000:])
    return d_target_pool, d_shadow_pool


def get_target_split(d_target_pool, train_size):
    """
    Returns (target_train, target_nonmember) — both size train_size.
    target_train  = members (used to train the target model)
    target_nonmember = non-members (held-out, used for balanced eval)
    """
    assert train_size <= len(d_target_pool) // 2, (
        f"train_size {train_size} too large for d_target_pool of size {len(d_target_pool)}"
    )
    indices = list(range(len(d_target_pool)))
    target_train     = Subset(d_target_pool, indices[:train_size])
    target_nonmember = Subset(d_target_pool, indices[train_size: train_size * 2])
    return target_train, target_nonmember


def get_shadow_split(d_shadow_pool, train_size, seed):
    """
    Returns (shadow_train, shadow_test) for one shadow model.
    Random subsample of d_shadow_pool; each shadow model gets a unique seed.
    """
    g = torch.Generator().manual_seed(seed)
    pool_size = len(d_shadow_pool)
    needed = train_size * 2
    assert needed <= pool_size, (
        f"Need {needed} examples but shadow pool only has {pool_size}"
    )
    perm = torch.randperm(pool_size, generator=g).tolist()
    shadow_train = Subset(d_shadow_pool, perm[:train_size])
    shadow_test  = Subset(d_shadow_pool, perm[train_size:needed])
    return shadow_train, shadow_test


def make_loader(dataset, batch_size=64, shuffle=False, num_workers=2):
    import torch
    pin = torch.cuda.is_available()  # only pin on CUDA, not MPS
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin)
