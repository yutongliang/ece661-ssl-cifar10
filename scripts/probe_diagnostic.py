#!/usr/bin/env python3
"""Recipe-free linear separability of frozen encoders (docs/RotNet.md, to-do 4).

The SGD linear evaluation of the late RotNet checkpoints hit its 100-epoch cap
still learning, so its accuracy mixes feature quality with optimization. This
script removes the optimizer from the question: for each encoder it extracts
features once (eval mode, no augmentation), standardizes them with training
statistics, and fits a multinomial logistic regression with full-batch L-BFGS
until convergence. The objective is convex, so the result depends only on the
features and the small L2 penalty.

Two feature sets per encoder: the pooled final stage (layer4, 2048-d, the one
the linear evaluation uses) and the pooled third stage (layer3, 1024-d), since
the RotNet paper found earlier blocks transfer better (its Table 1).

    python scripts/probe_diagnostic.py --out results/r50/diagnostics/lbfgs.csv \
        checkpoints/r50/RUN/encoder_epoch0300.pt checkpoints/r50/RUN/encoder_epoch1000.pt
"""
import argparse
import csv
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cifar_ssl import data  # noqa: E402
from cifar_ssl.models import load_encoder  # noqa: E402

FIELDS = ["checkpoint", "stage", "dim", "l2", "train_top1", "test_top1", "train_loss",
          "lbfgs_iters", "mean_sq_norm", "fraction_dead_dims", "seconds"]


@torch.no_grad()
def extract(encoder, train, device, batch_size=500):
    dataset = data.cifar10(train=train, transform=data.test_transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=8)
    feats = {3: [], 4: []}
    labels = []
    for x, y in loader:
        stages = encoder(x.to(device), return_stages=True)
        feats[3].append(F.adaptive_avg_pool2d(stages[2], 1).flatten(1).float())
        feats[4].append(F.adaptive_avg_pool2d(stages[3], 1).flatten(1).float())
        labels.append(y.to(device))
    return {k: torch.cat(v) for k, v in feats.items()}, torch.cat(labels)


def fit_logistic(x_train, y_train, x_test, y_test, l2, max_iter):
    w = torch.zeros(x_train.size(1), 10, device=x_train.device, requires_grad=True)
    b = torch.zeros(10, device=x_train.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=max_iter, history_size=50,
                            tolerance_grad=1e-6, tolerance_change=1e-9,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x_train @ w + b, y_train) + 0.5 * l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    iters = opt.state[opt._params[0]]["n_iter"]
    with torch.no_grad():
        train_loss = F.cross_entropy(x_train @ w + b, y_train).item()
        train_acc = ((x_train @ w + b).argmax(1) == y_train).float().mean().item()
        test_acc = ((x_test @ w + b).argmax(1) == y_test).float().mean().item()
    return train_acc, test_acc, train_loss, iters


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--out", required=True, help="CSV, appended; existing rows are skipped")
    p.add_argument("--l2", type=float, nargs="+", default=[1e-4, 1e-3])
    p.add_argument("--max-iter", type=int, default=1000)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r["checkpoint"], r["stage"], r["l2"]) for r in csv.DictReader(f)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for path in args.checkpoints:
            todo = [(s, l2) for s in (4, 3) for l2 in args.l2
                    if (path, f"layer{s}", f"{l2:g}") not in done]
            if not todo:
                print("Skip (done):", path)
                continue
            t0 = time.time()
            encoder, _ = load_encoder(path)
            encoder.to(device).eval()
            train_feats, y_train = extract(encoder, True, device)
            test_feats, y_test = extract(encoder, False, device)
            print(f"{path}: features in {time.time() - t0:.0f}s", flush=True)
            for stage, l2 in todo:
                xtr, xte = train_feats[stage], test_feats[stage]
                sq_norm = (xtr * xtr).sum(1).mean().item()
                dead = (xtr.std(0) < 1e-6).float().mean().item()
                mean, std = xtr.mean(0), xtr.std(0).clamp_min(1e-6)
                t1 = time.time()
                train_acc, test_acc, loss, iters = fit_logistic(
                    (xtr - mean) / std, y_train, (xte - mean) / std, y_test, l2, args.max_iter)
                row = {"checkpoint": path, "stage": f"layer{stage}", "dim": xtr.size(1),
                       "l2": f"{l2:g}", "train_top1": round(train_acc, 4),
                       "test_top1": round(test_acc, 4), "train_loss": round(loss, 4),
                       "lbfgs_iters": iters, "mean_sq_norm": round(sq_norm, 3),
                       "fraction_dead_dims": round(dead, 4),
                       "seconds": round(time.time() - t1, 1)}
                writer.writerow(row)
                f.flush()
                print(row, flush=True)


if __name__ == "__main__":
    main()
