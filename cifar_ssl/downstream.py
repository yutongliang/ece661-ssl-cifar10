"""Shared downstream recipe for the ResNet-50 comparison (docs/experiments.md).

One script for every downstream run, so all methods are trained the same way:

  fine-tune (1-10% labels):  --init ENCODER.pt --fraction 0.01
  supervised from scratch:    --arch resnet50 --fraction 0.01
  linear eval (100%, linear only): --init ENCODER.pt --freeze-encoder --fraction 1.0
  random-init linear control:      --arch resnet50 --freeze-encoder --fraction 1.0

Recipe: softmax cross-entropy, SGD with Nesterov momentum, no weight decay,
random crop and flip. The LR stays constant until training loss stops improving
by more than --plateau-threshold (relative) for --plateau-patience epochs; then
it is multiplied by --lr-factor. After --lr-drops such drops, the next plateau
stops training; --max-epochs is a hard cap. Only training loss decides when to
stop. Test top-1/top-3 is reported for the final model.

Each run owns --out-dir: last.pt (state saved every epoch, so a requeued Slurm
task resumes automatically) and result.json (written last, atomically). A run
whose result.json exists with the same settings is skipped.

    python -m cifar_ssl.downstream --init ENC.pt --fraction 0.01 --batch-size 256 \
        --lr 0.05 --out-dir results/r50/finetune/RUN/fraction0.01
"""
import argparse
import hashlib
import json
import math
import os
import socket
import tempfile
import time

import torch
import torch.nn as nn

from . import data
from .data import labeled_train_loader, test_loader
from .models import ARCHITECTURES, Classifier, build_encoder, checkpoint_arch, load_encoder
from .utils import (autocast_context, capture_rng_state, evaluate_topk, get_device,
                    restore_rng_state, save_checkpoint, set_seed)

# Settings that may change between attempts of the same run without changing results.
NON_IDENTITY_ARGS = {"num_workers", "out_dir", "eval_every", "keep_last"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--init", help="pretrained encoder checkpoint")
    source.add_argument("--arch", choices=ARCHITECTURES, help="random-init encoder architecture")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="train only the linear classifier (linear evaluation)")
    p.add_argument("--fraction", type=float, required=True, help="fraction of training labels")
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--lr", type=float, required=True, help="starting learning rate")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--plateau-patience", type=int, default=10)
    p.add_argument("--plateau-threshold", type=float, default=0.01,
                   help="relative training-loss improvement that counts as progress")
    p.add_argument("--lr-factor", type=float, default=0.1)
    p.add_argument("--lr-drops", type=int, default=1,
                   help="LR drops allowed; the plateau after the last drop stops training")
    p.add_argument("--min-epoch-size", type=int, default=5000,
                   help="minimum draws per epoch; small subsets sample with replacement")
    p.add_argument("--eval-every", type=int, default=0,
                   help="log test top-1 every N epochs for curves only (0 = final only)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--keep-last", action="store_true", help="keep last.pt after finishing")
    args = p.parse_args(argv)
    if not 0 < args.fraction <= 1:
        p.error("fraction must be in (0, 1]")
    if min(args.batch_size, args.max_epochs, args.plateau_patience) <= 0:
        p.error("batch size, max epochs, and patience must be positive")
    if not (args.lr > 0 and math.isfinite(args.lr)):
        p.error("learning rate must be finite and positive")
    if not 0 < args.lr_factor < 1 or args.lr_drops < 0 or args.plateau_threshold < 0:
        p.error("invalid plateau settings")
    return args


class PlateauRule:
    """Training-loss plateau detector: returns 'continue', 'drop', or 'stop' each epoch."""

    def __init__(self, patience, threshold, max_drops):
        self.patience, self.threshold, self.max_drops = patience, threshold, max_drops
        self.best, self.wait, self.drops = math.inf, 0, 0

    def step(self, loss):
        if loss < self.best * (1 - self.threshold):
            self.best, self.wait = loss, 0
            return "continue"
        self.wait += 1
        if self.wait < self.patience:
            return "continue"
        self.wait = 0
        if self.drops < self.max_drops:
            self.drops += 1
            return "drop"
        return "stop"

    def state_dict(self):
        return {"best": self.best, "wait": self.wait, "drops": self.drops}

    def load_state_dict(self, state):
        self.best, self.wait, self.drops = state["best"], state["wait"], state["drops"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(path) or ".", suffix=".tmp",
                                     delete=False) as f:
        json.dump(value, f, indent=2)
        temporary = f.name
    os.replace(temporary, path)


def run_identity(args):
    """Settings that define the run; attempts with different identities never mix."""
    identity = {k: v for k, v in vars(args).items() if k not in NON_IDENTITY_ARGS}
    if args.init:
        identity["init_sha256"] = sha256(args.init)
    if args.fraction < 1:
        subset = os.path.join(data.SUBSET_DIR,
                              f"cifar10_{args.fraction * 100:g}pct_seed{args.subset_seed}.json")
        if not os.path.isfile(subset):
            raise FileNotFoundError(f"Missing shared label subset {subset}; create it before training")
        identity["subset_sha256"] = sha256(subset)
    return identity


def build_model(args):
    if args.init:
        encoder, ckpt = load_encoder(args.init)
        method, arch = ckpt.get("method", "unknown"), checkpoint_arch(ckpt)
    else:
        encoder, ckpt = build_encoder(args.arch), {}
        method, arch = ("random_init" if args.freeze_encoder else "supervised"), args.arch
    model = Classifier(encoder)
    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
    return model, method, arch, ckpt


def train_epoch(model, loader, criterion, optimizer, scaler, device, frozen):
    model.train()
    if frozen:
        model.encoder.eval()  # frozen encoder: fixed batch-norm statistics
    total_loss = correct = seen = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with autocast_context(device):
            if frozen:
                with torch.no_grad():
                    features = model.encoder(inputs)
                outputs = model.fc(features.float())
            else:
                outputs = model(inputs)
            loss = criterion(outputs, targets)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * targets.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        seen += targets.size(0)
    return total_loss / seen, correct / seen


def main(argv=None):
    args = parse_args(argv)
    result_path = os.path.join(args.out_dir, "result.json")
    last_path = os.path.join(args.out_dir, "last.pt")
    identity = run_identity(args)
    if os.path.isfile(result_path):
        with open(result_path) as f:
            if json.load(f)["identity"] == identity:
                print(f"Already complete: {result_path}")
                return
        raise ValueError(f"{result_path} holds a different run; use a new --out-dir")

    set_seed(args.seed)
    device = get_device()
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True
    model, method, arch, ckpt = build_model(args)
    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=args.lr, momentum=args.momentum, nesterov=True,
                                weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss()  # standard softmax cross-entropy
    rule = PlateauRule(args.plateau_patience, args.plateau_threshold, args.lr_drops)

    start_epoch, history, elapsed = 1, [], 0.0
    if os.path.isfile(last_path):
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        if state["identity"] != identity or state["device"] != str(device):
            raise ValueError(f"{last_path} was written with different settings; use a new --out-dir")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        rule.load_state_dict(state["rule"])
        restore_rng_state(state["rng_state"], device)
        start_epoch, history, elapsed = state["epoch"] + 1, state["history"], state["seconds"]
        print(f"Resumed {last_path} at epoch {start_epoch}")

    # Loaders are built after any RNG restore so a resumed run draws the same batches.
    train_loader = labeled_train_loader(args.fraction, args.batch_size, seed=args.subset_seed,
                                        num_workers=args.num_workers,
                                        min_epoch_size=args.min_epoch_size)
    if not len(train_loader):
        raise ValueError("Batch size exceeds the epoch's draws; no complete batches")
    eval_loader = test_loader(num_workers=args.num_workers)
    experiment = ("linear_eval" if args.freeze_encoder else
                  "finetune" if args.init else "supervised")
    print(f"{experiment} | method={method} | arch={arch} | fraction={args.fraction} | "
          f"batch={args.batch_size} | lr={args.lr} | device={device} | "
          f"{len(train_loader)} steps/epoch")

    stop_reason = "max_epochs"
    epoch = start_epoch - 1
    already_stopped = bool(history) and history[-1]["decision"] == "stop"
    for epoch in range(start_epoch, 0 if already_stopped else args.max_epochs + 1):
        started = time.perf_counter()
        loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device,
                                      args.freeze_encoder)
        lr = optimizer.param_groups[0]["lr"]
        decision = rule.step(loss)
        if decision == "drop":
            for group in optimizer.param_groups:
                group["lr"] *= args.lr_factor
        elapsed += time.perf_counter() - started
        record = {"epoch": epoch, "loss": loss, "train_acc": train_acc, "lr": lr,
                  "decision": decision}
        msg = f"epoch {epoch:3d} | lr {lr:.4g} | loss {loss:.4f} | train acc {train_acc:.4f}"
        if args.eval_every and epoch % args.eval_every == 0:
            record["test_top1_monitor"] = evaluate_topk(model, eval_loader, device, ks=(1,))[1]
            msg += f" | test acc {record['test_top1_monitor']:.4f} (monitor only)"
        history.append(record)
        print(msg + ("" if decision == "continue" else f" | {decision}"))
        save_checkpoint(last_path, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "rule": rule.state_dict(), "epoch": epoch,
            "history": history, "seconds": elapsed, "identity": identity,
            "rng_state": capture_rng_state(device), "device": str(device),
        })
        if decision == "stop":
            break
    if history and history[-1]["decision"] == "stop":
        stop_reason, epoch = "plateau", history[-1]["epoch"]

    test = evaluate_topk(model, eval_loader, device, ks=(1, 3))
    print(f"final | stopped at epoch {epoch} ({stop_reason}) | "
          f"test top-1 {test[1]:.4f} | test top-3 {test[3]:.4f}")
    atomic_json(result_path, {
        "method": method, "experiment": experiment, "arch": arch,
        "label_fraction": args.fraction, "batch_size": args.batch_size, "lr": args.lr,
        "test_top1": test[1], "test_top3": test[3],
        "stop_epoch": epoch, "stop_reason": stop_reason, "lr_drops": rule.drops,
        "final_train_loss": history[-1]["loss"], "final_train_acc": history[-1]["train_acc"],
        "steps_per_epoch": len(train_loader), "train_seconds": elapsed,
        "init": args.init or "", "pretrain_args": ckpt.get("args", {}),
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if use_amp else "",
        "host": socket.gethostname(), "torch_version": str(torch.__version__),
        "identity": identity, "history": history,
    })
    if not args.keep_last and os.path.exists(last_path):
        os.remove(last_path)
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
