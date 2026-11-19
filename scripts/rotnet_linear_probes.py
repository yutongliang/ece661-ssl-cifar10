#!/usr/bin/env python3
"""Frozen linear probes (CIFAR-10 class top-1) for the RotNet Figure B.7 milestones.

Each probe reproduces `linear_eval.py` exactly (same seed order, feature
extraction, standardization and Adam/cosine linear classifier), but writes its
own JSON file instead of appending to the shared results/results.csv, so array
tasks can run concurrently.

    python scripts/rotnet_linear_probes.py run --task 0      # run_id #0: its 10 milestones
    python scripts/rotnet_linear_probes.py run --task 15     # random-init control
    python scripts/rotnet_linear_probes.py merge             # validate and build the CSV

Task indices 0-14 follow the order of runs in the milestone manifest; 15 is the
random-initialization floor. Completed probes whose checkpoint checksum and
protocol still match are skipped, so a requeued task resumes.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MANIFEST = os.path.join(ROOT, "experiments", "rotnet_figure_b7_milestones.json")
OUT_DIR = os.path.join(ROOT, "results", "rotnet_linear_probes")
MERGED_CSV = os.path.join(ROOT, "results", "rotnet_linear_probes.csv")
CONTROL_TASK = 15
PROTOCOL_KEYS = ("fraction", "stage", "pool", "probe_seed", "probe_epochs",
                 "probe_lr", "probe_weight_decay")


def load_manifest():
    with open(MANIFEST) as f:
        probes = json.load(f)["probes"]
    run_ids = list(dict.fromkeys(p["run_id"] for p in probes))
    return probes, run_ids


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(path), suffix=".tmp",
                                     delete=False) as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        temporary = f.name
    os.replace(temporary, path)


def result_path(run_id, epoch):
    return os.path.join(OUT_DIR, run_id, f"epoch{epoch:04d}.json")


def is_done(path, checkpoint_sha, probe):
    if not os.path.exists(path):
        return False
    with open(path) as f:
        previous = json.load(f)
    return (previous.get("checkpoint_sha256") == checkpoint_sha
            and all(previous.get(k) == probe[k] for k in PROTOCOL_KEYS))


def fit_probe(make_encoder, train_set, test_set, probe, device, num_workers):
    """Same computation and RNG order as linear_eval.main().

    linear_eval seeds before building the encoder (ResNet20 init consumes RNG
    even when weights are then loaded), so `make_encoder` is called after seeding.
    """
    import torch
    import torch.nn as nn
    from linear_eval import extract_features
    from utils import set_seed

    set_seed(probe["probe_seed"])
    encoder, extra = make_encoder()
    encoder = encoder.to(device)
    stage, pool = probe["stage"], probe["pool"]
    x_train, y_train = extract_features(encoder, train_set, stage, pool, device, num_workers)
    x_test, y_test = extract_features(encoder, test_set, stage, pool, device, num_workers)
    mean, std = x_train.mean(0), x_train.std(0) + 1e-6
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    classifier = nn.Linear(x_train.size(1), 10).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=probe["probe_lr"],
                                 weight_decay=probe["probe_weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=probe["probe_epochs"])
    criterion = nn.CrossEntropyLoss()
    batch_size = 256
    for _ in range(probe["probe_epochs"]):
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
    return extra, {"train_acc": train_acc, "test_acc": test_acc, "feature_dim": x_train.size(1),
            "num_train": x_train.size(0), "num_test": x_test.size(0),
            "probe_batch_size": batch_size, "probe_optimizer": "Adam",
            "probe_steps_per_epoch": (x_train.size(0) + batch_size - 1) // batch_size}


def run(task, num_workers):
    from importlib.metadata import version

    import torch
    from torch.utils.data import Subset

    from data import cifar10, label_subset_indices, test_transform
    from models import ResNet20, load_encoder
    from utils import get_device

    probes, run_ids = load_manifest()
    if task == CONTROL_TASK:
        template = dict(probes[0])
        for key in ("run_id", "global_batch_size", "lr_slot", "lr", "pretrain_seed",
                    "checkpoint_epoch", "checkpoint"):
            template[key] = None
        template.update(run_id="random_init", checkpoint_epoch=0)
        selected = [template]
    elif 0 <= task < len(run_ids):
        selected = [p for p in probes if p["run_id"] == run_ids[task]]
    else:
        raise SystemExit(f"--task must be 0-{len(run_ids) - 1} or {CONTROL_TASK}")

    device = get_device()
    if os.environ.get("SLURM_JOB_ID") and device.type != "cuda":
        raise SystemExit("CUDA unavailable inside a Slurm job")
    print(f"Task {task}: {len(selected)} probes | device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)

    fractions = {p["fraction"] for p in selected}
    if fractions != {1.0}:
        raise SystemExit(f"Unexpected label fractions {fractions}; this driver handles 100% only")
    full_train = cifar10(train=True, transform=test_transform)
    train_set = Subset(full_train, label_subset_indices(full_train.targets, 1.0, 0))
    test_set = cifar10(train=False, transform=test_transform)

    for probe in selected:
        run_id, epoch = probe["run_id"], probe["checkpoint_epoch"]
        out = result_path(run_id, epoch)
        if probe["checkpoint"]:
            ckpt_path = os.path.join(ROOT, probe["checkpoint"])
            checkpoint_sha = sha256(ckpt_path)
        else:
            ckpt_path, checkpoint_sha = None, "random_init"
        if is_done(out, checkpoint_sha, probe):
            print(f"skip {run_id} epoch {epoch}: already complete", flush=True)
            continue
        start = time.time()
        if ckpt_path:
            def make_encoder(path=ckpt_path, expected_lr=probe["lr"]):
                encoder, ckpt = load_encoder(path)
                args = ckpt.get("args", {})
                if args.get("lr") is not None and float(args["lr"]) != float(expected_lr):
                    raise SystemExit(f"{path}: checkpoint lr {args['lr']} != manifest {expected_lr}")
                return encoder, args
        else:
            def make_encoder():
                return ResNet20(), {}
        pretrain_args, metrics = fit_probe(make_encoder, train_set, test_set, probe, device,
                                           num_workers)
        record = {**probe, **metrics, "checkpoint_sha256": checkpoint_sha,
                  "pretrain_args": pretrain_args, "seconds": round(time.time() - start, 2),
                  "device": str(device),
                  "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                  "torch_version": str(torch.__version__),
                  "torchvision_version": version("torchvision"),
                  "slurm_job": os.environ.get("SLURM_JOB_ID"),
                  "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "status": "complete"}
        write_json_atomic(out, record)
        print(f"{run_id} epoch {epoch:4d} | class top-1 test {metrics['test_acc']:.4f} | "
              f"train {metrics['train_acc']:.4f} | {record['seconds']:.0f}s", flush=True)


def merge():
    probes, _ = load_manifest()
    rows, missing = [], []
    for probe in probes + [{"run_id": "random_init", "checkpoint_epoch": 0}]:
        path = result_path(probe["run_id"], probe["checkpoint_epoch"])
        if not os.path.exists(path):
            missing.append(path)
            continue
        with open(path) as f:
            rows.append(json.load(f))
    fields = ["run_id", "global_batch_size", "lr", "checkpoint_epoch", "test_acc", "train_acc",
              "fraction", "stage", "pool", "probe_seed", "probe_epochs", "probe_lr",
              "probe_weight_decay", "checkpoint", "checkpoint_sha256", "seconds", "gpu",
              "slurm_job", "finished_utc"]
    os.makedirs(os.path.dirname(MERGED_CSV), exist_ok=True)
    with open(MERGED_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {MERGED_CSV}")
    if missing:
        print(f"MISSING {len(missing)} probes:", *missing, sep="\n  ")
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--task", type=int, required=True)
    run_parser.add_argument("--num-workers", type=int, default=4)
    sub.add_parser("merge")
    args = parser.parse_args()
    if args.command == "run":
        run(args.task, args.num_workers)
    else:
        merge()


if __name__ == "__main__":
    main()
