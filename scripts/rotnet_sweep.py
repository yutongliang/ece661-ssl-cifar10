#!/usr/bin/env python3
"""Validate or run one of the 15 RotNet configurations. No fine-tuning here."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from shared_lock import directory_lock

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/rotnet_figure_b7_sweep.csv"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_runs(path=MANIFEST):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 15:
        raise ValueError("RotNet grid must contain exactly 15 configurations")
    expected = {(batch, rate) for batch in (256, 512, 1024, 2048, 4096)
                for rate in (0.03, 0.1, 0.3)}
    seen = set()
    paths = set()
    for row in rows:
        batch, rate = int(row["global_batch_size"]), float(row["lr"])
        pair = (batch, rate)
        directory = Path(row["checkpoint_dir"])
        if (pair not in expected or pair in seen or row["method"] != "rotnet"
                or row["backbone"] != "ResNet20" or int(row["epochs"]) != 1000
                or int(row["pretrain_seed"]) != 0 or row["lr_scaling"] != "constant"
                or directory.is_absolute() or ".." in directory.parts
                or str(directory) in paths or row["run_id"] != directory.name
                or int(row["encoder_every_epochs"]) != 100
                or int(row["steps_per_epoch"]) != 50000 // batch
                or float(row["weight_decay"]) != 5e-4):
            raise ValueError(f"Invalid or duplicate grid row: {row}")
        seen.add(pair)
        paths.add(str(directory))
    if seen != expected:
        raise ValueError("Grid does not contain all five batches and three rates")
    return rows


def training_arguments(row, output):
    return ["--epochs", row["epochs"], "--batch-size", row["global_batch_size"],
            "--lr", row["lr"], "--weight-decay", row["weight_decay"],
            "--seed", row["pretrain_seed"], "--num-workers", "4",
            "--eval-every", "100", "--checkpoint-every", "100",
            "--out", str(output), "--require-cuda"]


def source_hash():
    digest = hashlib.sha256()
    for name in ("pretrain_rotnet.py", "models.py", "data.py", "utils.py"):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def run_one(row, phase, pilot_epochs):
    import torch

    directory = ROOT / row["checkpoint_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "encoder.pt"
    last = Path(str(output) + ".last")
    identity = {"configuration": row, "training_source_sha256": source_hash()}
    with directory_lock(directory / ".run.lock.d"):
        metadata = directory / "run.json"
        if metadata.exists():
            if json.loads(metadata.read_text()) != identity:
                raise ValueError("Saved configuration/source differs; use a new run directory")
        else:
            if last.exists() or output.exists():
                raise ValueError("Existing weights have no matching run metadata; refusing overwrite")
            atomic_json(metadata, identity)
        if phase == "train" and not (directory / "pilot_pass.json").is_file():
            raise ValueError("This configuration has not passed its GPU pilot")
        state = torch.load(last, map_location="cpu", weights_only=True) if last.exists() else None
        current = state["epoch"] if state is not None else 0
        target = pilot_epochs if phase == "pilot" else int(row["epochs"])
        if current < target:
            command = [sys.executable, "-u", str(ROOT / "pretrain_rotnet.py"),
                       *training_arguments(row, output)]
            if last.exists():
                command.append("--resume")
            if phase == "pilot":
                command.extend(["--stop-after-epoch", str(pilot_epochs)])
            print(f"{phase}: {row['run_id']} | B={row['global_batch_size']} "
                  f"({4 * int(row['global_batch_size'])} rotated images) | lr={row['lr']}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            state = torch.load(last, map_location="cpu", weights_only=True)
        if state is None or state["epoch"] < target:
            raise RuntimeError("Training did not reach its requested epoch")
        if state["args"]["epochs"] != 1000 or state["scheduler"]["T_max"] != 1000:
            raise RuntimeError("The original 1000-epoch schedule was not preserved")
        if phase == "pilot":
            history = [item for item in state["history"] if item["epoch"] <= pilot_epochs]
            if len(history) != pilot_epochs or any(not math.isfinite(item["loss"]) for item in history):
                raise RuntimeError("Pilot history is missing or has non-finite losses")
            timed = history[1:] or history
            mean_seconds = sum(item["seconds"] for item in timed) / len(timed)
            report = {"run_id": row["run_id"], "epochs_checked": pilot_epochs,
                      "schedule_epochs": 1000, "seconds_per_epoch": mean_seconds,
                      "estimated_1000_epoch_hours": mean_seconds * 1000 / 3600,
                      "peak_reserved_GiB": max(item["peak_reserved_bytes"] for item in history) / 2**30,
                      "torch_version": state["torch_version"], "device": state["device"]}
            atomic_json(directory / "pilot_pass.json", report)
            print(json.dumps(report), flush=True)
        else:
            if not output.is_file():
                raise RuntimeError("Final encoder is missing")
            final = torch.load(output, map_location="cpu", weights_only=True)
            if final.get("epochs") != 1000 or final.get("method") != "rotnet":
                raise RuntimeError("Final encoder metadata is inconsistent")
            for epoch in range(100, 1001, 100):
                milestone = directory / f"encoder_epoch{epoch:04d}.pt"
                if not milestone.is_file():
                    raise RuntimeError(f"Missing milestone: {milestone}")
                saved = torch.load(milestone, map_location="cpu", weights_only=True)
                if saved.get("epochs") != epoch:
                    raise RuntimeError(f"Wrong milestone epoch: {milestone}")
            atomic_json(directory / "complete.json", {**identity, "epochs": 1000,
                        "encoder_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
            print(f"Completed and verified: {row['run_id']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "preflight", "run", "pilots"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--phase", choices=("pilot", "train"), default="train")
    parser.add_argument("--pilot-epochs", type=int, default=5)
    args = parser.parse_args()
    rows = read_runs()
    if not 1 <= args.pilot_epochs < 100:
        parser.error("pilot-epochs must be between 1 and 99")
    if args.action == "validate":
        for index, row in enumerate(rows):
            print(index, row["run_id"], "B=" + row["global_batch_size"], "lr=" + row["lr"])
    elif args.action == "preflight":
        sys.path.insert(0, str(ROOT))
        import torch
        import torchvision
        print("Environment:", torch.__version__, torchvision.__version__, flush=True)
        if (str(torch.__version__).split("+", 1)[0] != "2.10.0"
                or str(torchvision.__version__).split("+", 1)[0] != "0.25.0"):
            raise RuntimeError("Use the pinned torch 2.10.0 / torchvision 0.25.0 environment")
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                       cwd=ROOT, check=True)
        from data import cifar10, test_transform
        for train in (True, False):
            dataset = cifar10(train=train, transform=test_transform)
            print("Prepared CIFAR-10:", "train" if train else "test", len(dataset))
    else:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Sweep requires a usable CUDA GPU")
        print("GPU:", torch.cuda.get_device_name(0), "torch:", torch.__version__, flush=True)
        if args.action == "pilots":
            # Fail early if the largest physical batch or highest rate is invalid.
            for row in sorted(rows, key=lambda item: (int(item["global_batch_size"]),
                                                       float(item["lr"])), reverse=True):
                run_one(row, "pilot", args.pilot_epochs)
            for row in rows:
                report = json.loads((ROOT / row["checkpoint_dir"] / "pilot_pass.json").read_text())
                if report["estimated_1000_epoch_hours"] * 1.25 > 48:
                    raise RuntimeError(f"{row['run_id']} needs more than the 48-hour request "
                                       "with a 25% timing margin; revise scheduling before full training")
        else:
            if args.index is None or not 0 <= args.index < len(rows):
                parser.error("run requires an index from 0 to 14")
            run_one(rows[args.index], args.phase, args.pilot_epochs)


if __name__ == "__main__":
    main()
