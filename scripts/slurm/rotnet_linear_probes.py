#!/usr/bin/env python3
"""Frozen linear probes (CIFAR-10 class top-1) for the RotNet Figure B.7 milestones.

Probes share the classifier fit with linear_eval.py. Each task writes one JSON
per checkpoint so array tasks can run concurrently.

    python scripts/slurm/rotnet_linear_probes.py run --task 0      # run_id #0: its 10 milestones
    python scripts/slurm/rotnet_linear_probes.py run --task 15     # random-init control
    python scripts/slurm/rotnet_linear_probes.py merge             # validate and build the CSV

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
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)

MANIFEST = os.path.join(ROOT, "experiments", "rotnet", "milestones.json")
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


def run(task, num_workers):
    from importlib.metadata import version

    import torch
    from torch.utils.data import Subset

    from cifar_ssl.data import cifar10, label_subset_indices, test_transform
    from cifar_ssl.linear_eval import fit_probe
    from cifar_ssl.models import ResNet20, load_encoder
    from cifar_ssl.utils import get_device, set_seed

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
        set_seed(probe["probe_seed"])
        if ckpt_path:
            encoder, ckpt = load_encoder(ckpt_path)
            pretrain_args = ckpt.get("args", {})
            stored_lr, expected_lr = pretrain_args.get("lr"), probe["lr"]
            if stored_lr is not None and float(stored_lr) != float(expected_lr):
                raise SystemExit(f"{ckpt_path}: checkpoint lr {stored_lr} != manifest {expected_lr}")
        else:
            encoder, pretrain_args = ResNet20(), {}
        metrics = fit_probe(encoder, train_set, test_set, stage=probe["stage"],
                            pool=probe["pool"], device=device, num_workers=num_workers,
                            epochs=probe["probe_epochs"], lr=probe["probe_lr"],
                            weight_decay=probe["probe_weight_decay"])
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
