"""CIFAR-10 loading and the fixed label subsets shared by every experiment.

Label subsets are class-balanced (1% = 50 images per class) and saved to
subsets/ as JSON the first time they are made, so SimCLR, RotNet, SimSiam
and the supervised baseline all train on exactly the same images.
"""
import json
import os

import numpy as np
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_DIR, "data")
SUBSET_DIR = os.path.join(PROJECT_DIR, "subsets")

# Standard supervised augmentation (same as HW2)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def cifar10(train, transform, root=DEFAULT_DATA_ROOT):
    return datasets.CIFAR10(root=root, train=train, transform=transform, download=True)


def make_label_subset(targets, fraction, seed=0, num_classes=10):
    """Pick `fraction` of the training set with an equal number per class."""
    targets = np.asarray(targets)
    per_class = int(round(len(targets) * fraction / num_classes))
    rng = np.random.RandomState(seed)
    indices = []
    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        indices.extend(rng.choice(class_indices, per_class, replace=False).tolist())
    return sorted(indices)


def label_subset_indices(targets, fraction, seed=0):
    """Load the saved subset for (fraction, seed), creating it if needed.

    fraction=1.0 returns every index and is not saved.
    """
    if fraction >= 1.0:
        return list(range(len(targets)))
    os.makedirs(SUBSET_DIR, exist_ok=True)
    path = os.path.join(SUBSET_DIR, f"cifar10_{fraction * 100:g}pct_seed{seed}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["indices"]
    indices = make_label_subset(targets, fraction, seed)
    with open(path, "w") as f:
        json.dump({"fraction": fraction, "seed": seed, "indices": indices}, f)
    print(f"Saved new label subset ({len(indices)} images) to {path}")
    return indices


def labeled_train_loader(fraction, batch_size, seed=0, transform=train_transform,
                         min_epoch_size=5000, num_workers=4, root=DEFAULT_DATA_ROOT):
    """Loader over the labeled subset.

    Small subsets (1% = 500 images) would only give a few steps per epoch, so
    each epoch samples at least `min_epoch_size` images with replacement.
    """
    full = cifar10(train=True, transform=transform, root=root)
    subset = Subset(full, label_subset_indices(full.targets, fraction, seed))
    if len(subset) < min_epoch_size:
        sampler = RandomSampler(subset, replacement=True, num_samples=min_epoch_size)
        return DataLoader(subset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, drop_last=True)
    return DataLoader(subset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, drop_last=True)


def test_loader(batch_size=512, num_workers=4, root=DEFAULT_DATA_ROOT):
    return DataLoader(cifar10(train=False, transform=test_transform, root=root),
                      batch_size=batch_size, shuffle=False, num_workers=num_workers)
