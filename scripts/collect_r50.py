#!/usr/bin/env python3
"""Merge ResNet-50 downstream result.json files into one CSV per experiment.

    python scripts/collect_r50.py [--root results/r50]

Writes <root>/linear.csv and <root>/finetune.csv (accuracies as fractions) and
lists expected runs that have no result yet. Exits 1 if any are missing.
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = ["rotnet_r50_b256_lr0p1_seed0", "rotnet_r50_b1024_lr0p4_seed0"]
EXPECTED = {
    "linear": [f"{run}/epoch{e:04d}" for run in RUNS for e in range(100, 1001, 100)] + ["random_init"],
    "finetune": [f"{run}/fraction{f}" for run in RUNS for f in ("0.01", "0.03", "0.05", "0.1")],
}
FIELDS = ["run", "method", "experiment", "arch", "label_fraction", "batch_size", "lr",
          "test_top1", "test_top3", "stop_epoch", "stop_reason", "lr_drops",
          "final_train_loss", "final_train_acc", "steps_per_epoch", "train_seconds", "gpu", "init"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(ROOT, "results", "r50"))
    args = parser.parse_args()
    missing = []
    for experiment, runs in EXPECTED.items():
        rows = []
        for run in runs:
            path = os.path.join(args.root, experiment, run, "result.json")
            if not os.path.isfile(path):
                missing.append(path)
                continue
            with open(path) as f:
                result = json.load(f)
            rows.append({"run": run, **{k: result.get(k, "") for k in FIELDS if k != "run"}})
        if rows:
            out = os.path.join(args.root, f"{experiment}.csv")
            with open(out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            print(f"Wrote {len(rows)} rows to {out}")
    if missing:
        print(f"MISSING {len(missing)} results:", *missing, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
