"""Helpers shared by all training scripts."""
import csv
import json
import os
import random
import tempfile
from contextlib import nullcontext
from datetime import datetime
from importlib.metadata import version

import numpy as np
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(PROJECT_DIR, "results", "results.csv")
RESULT_FIELDS = ["timestamp", "method", "experiment", "label_fraction", "stage",
                 "test_acc", "test_top3", "epochs", "batch_size", "lr", "seed", "checkpoint",
                 "subset_seed", "weight_decay", "pool", "optimizer", "steps_per_epoch",
                 "min_epoch_size", "device", "torch_version", "torchvision_version",
                 "config", "pretrain_config"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device):
    """Use mixed precision only on CUDA; CPU and MPS run in float32."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def cap_gpu_memory(gigabytes, device):
    """Limit this process to `gigabytes` of GPU memory, to emulate a smaller GPU."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    total = torch.cuda.get_device_properties(index).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gigabytes * 2**30 / total), index)
    print(f"GPU memory capped at {gigabytes:g} GiB of {total / 2**30:.1f} GiB")


def capture_rng_state(device):
    """Store RNG state using tensors and primitives accepted by weights_only loading."""
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state, device):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    elif device.type == "mps":
        torch.mps.set_rng_state(state["mps"].cpu())


def save_checkpoint(path, state):
    """Create the directory and replace a checkpoint only after writing succeeds."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as f:
        temporary = f.name
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


@torch.no_grad()
def evaluate(model, loader, device):
    """Top-1 accuracy of a classifier on a loader."""
    return evaluate_topk(model, loader, device, ks=(1,))[1]


def topk_correct(logits, targets, ks=(1, 3)):
    """Number of samples whose label is among the k highest logits, for each k."""
    top = logits.topk(max(ks), dim=1).indices
    hits = top == targets.unsqueeze(1)
    return {k: hits[:, :k].any(1).sum().item() for k in ks}


@torch.no_grad()
def evaluate_topk(model, loader, device, ks=(1, 3)):
    """Top-k accuracies {k: acc} of a classifier on a loader."""
    model.eval()
    correct, total = dict.fromkeys(ks, 0), 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        for k, c in topk_correct(model(inputs), targets, ks).items():
            correct[k] += c
        total += targets.size(0)
    return {k: c / total for k, c in correct.items()}


def save_encoder(path, encoder, method, args, **extra):
    """Save only what downstream evaluation needs: the encoder weights + settings."""
    save_checkpoint(path, {"encoder": encoder.state_dict(), "method": method,
                           "args": vars(args), **extra})


def log_result(**row):
    """Append one row to results/results.csv (one shared table for the report)."""
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    new_file = not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
    if not new_file:
        with open(RESULTS_CSV, newline="") as f:
            reader = csv.DictReader(f)
            previous_fields = reader.fieldnames
            previous_rows = list(reader)
        if previous_fields != RESULT_FIELDS:
            if not set(previous_fields).issubset(RESULT_FIELDS):
                raise ValueError("Results CSV has unknown columns; refusing to discard them")
            # Keep older results, leaving unavailable metadata blank.
            with tempfile.NamedTemporaryFile(mode="w", dir=os.path.dirname(RESULTS_CSV),
                                             newline="", delete=False) as f:
                temporary = f.name
                try:
                    writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
                    writer.writeheader()
                    writer.writerows(previous_rows)
                except BaseException:
                    os.remove(temporary)
                    raise
            try:
                os.replace(temporary, RESULTS_CSV)
            finally:
                if os.path.exists(temporary):
                    os.remove(temporary)
    row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    row.setdefault("torch_version", str(torch.__version__))
    row.setdefault("torchvision_version", version("torchvision"))
    for key in ("config", "pretrain_config"):
        if isinstance(row.get(key), dict):
            row[key] = json.dumps(row[key], sort_keys=True)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print("Logged:", {k: row.get(k) for k in RESULT_FIELDS if row.get(k) is not None})
