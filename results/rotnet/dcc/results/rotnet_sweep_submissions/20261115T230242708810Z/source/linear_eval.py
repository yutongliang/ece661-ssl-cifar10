"""Linear evaluation: freeze a pretrained encoder, train only a linear classifier.

Features are extracted once (no augmentation), then a linear layer is trained
on them, which takes seconds instead of re-running the encoder every epoch.

    python linear_eval.py --checkpoint checkpoints/rotnet.pt
    python linear_eval.py --checkpoint checkpoints/rotnet.pt --stage 2 --pool 2   # layer-wise
    python linear_eval.py --random-init                                         # sanity floor
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data import cifar10, label_subset_indices, test_transform
from models import ResNet20, load_encoder
from utils import get_device, log_result, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--random-init", action="store_true", help="evaluate an untrained encoder")
    p.add_argument("--stage", type=int, default=3, choices=[1, 2, 3],
                   help="which ResNet-20 stage to take features from (3 = final)")
    p.add_argument("--pool", type=int, default=1,
                   help="spatial size after average pooling (1 -> 64-d for stage 3; 2 -> 4x larger)")
    p.add_argument("--fraction", type=float, default=1.0, help="fraction of labels for the classifier")
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def extract_features(encoder, dataset, stage, pool, device, num_workers):
    encoder.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=num_workers)
    features, labels = [], []
    for images, targets in loader:
        stage_map = encoder(images.to(device), return_stages=True)[stage - 1]
        features.append(F.adaptive_avg_pool2d(stage_map, pool).flatten(1).cpu())
        labels.append(targets)
    return torch.cat(features), torch.cat(labels)


def main():
    args = parse_args()
    if not args.checkpoint and not args.random_init:
        raise SystemExit("Pass --checkpoint PATH or --random-init")
    set_seed(args.seed)
    device = get_device()

    if args.random_init:
        encoder, method, ckpt_path = ResNet20(), "random_init", ""
        ckpt = {}
    else:
        encoder, ckpt = load_encoder(args.checkpoint)
        method, ckpt_path = ckpt["method"], args.checkpoint
    encoder = encoder.to(device)

    full_train = cifar10(train=True, transform=test_transform)
    train_set = Subset(full_train, label_subset_indices(full_train.targets, args.fraction,
                                                        args.subset_seed))
    test_set = cifar10(train=False, transform=test_transform)

    x_train, y_train = extract_features(encoder, train_set, args.stage, args.pool, device, args.num_workers)
    x_test, y_test = extract_features(encoder, test_set, args.stage, args.pool, device, args.num_workers)

    # Standardize with training statistics so one learning rate works for every stage.
    mean, std = x_train.mean(0), x_train.std(0) + 1e-6
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)
    print(f"{method} | stage {args.stage} | feature dim {x_train.size(1)} | "
          f"{x_train.size(0)} labeled train images")

    classifier = nn.Linear(x_train.size(1), 10).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    batch_size = 256

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        perm = torch.randperm(x_train.size(0), device=device)
        for i in range(0, x_train.size(0), batch_size):
            idx = perm[i:i + batch_size]
            loss = criterion(classifier(x_train[idx]), y_train[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()

    classifier.eval()
    with torch.no_grad():
        train_acc = (classifier(x_train).argmax(1) == y_train).float().mean().item()
        test_acc = (classifier(x_test).argmax(1) == y_test).float().mean().item()
    print(f"linear eval | train acc {train_acc:.4f} | test acc {test_acc:.4f}")

    log_result(method=method, experiment=f"linear_eval_pool{args.pool}", label_fraction=args.fraction,
               stage=args.stage, test_acc=round(test_acc, 4), epochs=args.epochs,
               lr=args.lr, seed=args.seed, checkpoint=ckpt_path, batch_size=batch_size,
               subset_seed=args.subset_seed, weight_decay=args.weight_decay, pool=args.pool,
               optimizer="Adam", steps_per_epoch=(x_train.size(0) + batch_size - 1) // batch_size,
               device=str(device), config=vars(args), pretrain_config=ckpt.get("args", {}))


if __name__ == "__main__":
    main()
