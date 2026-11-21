"""RotNet pretraining (Gidaris et al., ICLR 2018) on CIFAR-10, no class labels.

Every image is shown at 0/90/180/270 degrees and the network predicts which
rotation was applied (4-way classification). After training, only the encoder
is kept; the rotation head is thrown away.

    python -m cifar_ssl.pretrain_rotnet --epochs 100
    python -m cifar_ssl.pretrain_rotnet --epochs 100 --resume   # continue after a disconnect
"""
import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from .data import cifar10, test_transform, train_transform
from .models import ARCHITECTURES, build_encoder
from .utils import (autocast_context, capture_rng_state, get_device, restore_rng_state,
                    save_checkpoint, save_encoder, set_seed)

NUM_ROTATIONS = 4
# Settings added after the ResNet-20 sweep; older resume states used these values.
LEGACY_DEFAULTS = {"arch": "resnet20", "accum_steps": 1}


def load_resume_state(args, device):
    """Resume the same planned run, never silently change its training schedule."""
    path = args.out + ".last"
    if not os.path.isfile(path):
        raise ValueError(f"Cannot resume: checkpoint {path} does not exist")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not {"args", "rng_state", "device", "torch_version"}.issubset(state):
        raise ValueError("Legacy resume checkpoint lacks settings/RNG state. Start a new run "
                         "with a new --out path; existing final encoders remain usable.")
    saved = {**LEGACY_DEFAULTS, **state["args"]}
    differences = [f"{key}: saved={saved.get(key)!r}, requested={value!r}"
                   for key, value in vars(args).items()
                   if key not in {"out", "resume", "stop_after_epoch", "checkpoint_every",
                                  "require_cuda"}
                   and saved.get(key) != value]
    if state["device"] != str(device):
        differences.append(f"device: saved={state['device']}, requested={device}")
    if state["torch_version"] != str(torch.__version__):
        differences.append("PyTorch version changed")
    if differences:
        raise ValueError("Resume settings differ: " + "; ".join(differences) +
                         ". Reuse the original settings, including the total --epochs.")
    return state


def rotate_batch(images):
    """(B,C,H,W) -> (4B,C,H,W) rotated copies and their rotation labels 0..3."""
    rotated = [torch.rot90(images, k, dims=(2, 3)) for k in range(NUM_ROTATIONS)]
    labels = torch.arange(NUM_ROTATIONS).repeat_interleave(images.size(0))
    return torch.cat(rotated), labels


def rotation_step(encoder, head, criterion, images, optimizer, scaler, device, accum_steps=1):
    """One optimizer update on a batch of original images and their four rotations.

    With accum_steps > 1 the batch is split into equal sub-batches whose gradients
    are summed before the single update. The gradient matches the full batch, but
    batch norm normalizes each sub-batch separately. Returns (loss sum, correct, count).
    """
    if images.size(0) % accum_steps:
        raise ValueError("Batch size must be divisible by accum_steps")
    optimizer.zero_grad(set_to_none=True)
    loss_sum = correct = count = 0
    for chunk in images.chunk(accum_steps):
        rotated, labels = rotate_batch(chunk)
        rotated, labels = rotated.to(device), labels.to(device)
        with autocast_context(device):
            logits = head(encoder(rotated))
            loss = criterion(logits, labels)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite rotation loss")
        scaler.scale(loss / accum_steps).backward()
        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        count += labels.size(0)
    scaler.step(optimizer)
    scaler.update()
    return loss_sum, correct, count


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=ARCHITECTURES, default="resnet20")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128, help="images per batch (x4 after rotation)")
    p.add_argument("--accum-steps", type=int, default=1,
                   help="split each batch into this many sub-batches and accumulate gradients")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--out", default="checkpoints/rotnet.pt", help="final encoder checkpoint")
    p.add_argument("--resume", action="store_true", help="resume from <out>.last")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="export encoder_epochNNNN.pt beside --out every N epochs (0 disables)")
    p.add_argument("--stop-after-epoch", type=int, default=None,
                   help="stop after this absolute epoch without changing the total schedule")
    p.add_argument("--require-cuda", action="store_true", help="fail rather than train on CPU/MPS")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.eval_every) <= 0:
        p.error("epochs, batch size, and evaluation interval must be positive")
    if args.lr <= 0 or not math.isfinite(args.lr):
        p.error("learning rate must be finite and positive")
    if args.accum_steps < 1 or args.batch_size % args.accum_steps:
        p.error("accum-steps must be positive and divide the batch size")
    if args.checkpoint_every < 0:
        p.error("checkpoint interval cannot be negative")
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= args.epochs:
        p.error("stop-after-epoch must be between 1 and the total epochs")
    return args


@torch.no_grad()
def rotation_accuracy(encoder, head, loader, device):
    encoder.eval()
    head.eval()
    correct = total = 0
    for images, _ in loader:  # class labels are ignored
        images, labels = rotate_batch(images)
        images, labels = images.to(device), labels.to(device)
        correct += (head(encoder(images)).argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this run; refusing CPU/MPS fallback")
    use_amp = device.type == "cuda"
    if use_amp and args.arch == "resnet50":
        torch.backends.cudnn.benchmark = True  # fixed input shapes; faster ResNet-50 kernels
    # Validate before downloading data or constructing a new training run.
    state = load_resume_state(args, device) if args.resume else None

    # Class labels come with the dataset but are never used for training.
    train_loader = torch.utils.data.DataLoader(
        cifar10(train=True, transform=train_transform), batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, drop_last=True)
    if not len(train_loader):
        raise ValueError("Batch size exceeds the training set; there are no complete batches")
    test_loader = torch.utils.data.DataLoader(
        cifar10(train=False, transform=test_transform), batch_size=256,
        shuffle=False, num_workers=args.num_workers)

    encoder = build_encoder(args.arch).to(device)
    head = nn.Linear(encoder.feature_dim, NUM_ROTATIONS).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss()

    # Full training state, saved every epoch so a lost session can resume.
    last_path = args.out + ".last"
    start_epoch = 1
    history = []
    if state is not None:
        encoder.load_state_dict(state["encoder"])
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        # Restore after model initialization, before creating any loader iterator.
        restore_rng_state(state["rng_state"], device)
        start_epoch = state["epoch"] + 1
        history = state.get("history", [])
        print(f"Resumed from {last_path} at epoch {start_epoch}")

    print(f"RotNet pretraining | arch={args.arch} | device={device} | "
          f"{len(train_loader)} steps/epoch | {args.accum_steps} sub-batches per step")
    end_epoch = args.stop_after_epoch or args.epochs
    completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, end_epoch + 1):
        if use_amp:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        encoder.train()
        head.train()
        total_loss = correct = seen = 0
        for images, _ in train_loader:
            try:
                step_loss, step_correct, step_seen = rotation_step(
                    encoder, head, criterion, images, optimizer, scaler, device, args.accum_steps)
            except FloatingPointError as error:
                raise FloatingPointError(f"{error} at epoch {epoch}") from None
            total_loss += step_loss
            correct += step_correct
            seen += step_seen
        scheduler.step()

        msg = f"epoch {epoch:3d} | loss {total_loss / seen:.4f} | rotation acc {correct / seen:.4f}"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            msg += f" | test rotation acc {rotation_accuracy(encoder, head, test_loader, device):.4f}"
        if use_amp:
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        peak_bytes = torch.cuda.max_memory_reserved(device) if use_amp else 0
        msg += f" | seconds {seconds:.2f} | peak reserved GiB {peak_bytes / 2**30:.2f}"
        print(msg)

        history.append({"epoch": epoch, "loss": total_loss / seen,
                        "rotation_acc": correct / seen, "seconds": seconds,
                        "peak_reserved_bytes": peak_bytes})
        # Publish milestones before the last-state commit, so resuming after an
        # interrupted export repeats the epoch rather than silently losing a snapshot.
        if args.checkpoint_every and (epoch % args.checkpoint_every == 0 or epoch == args.epochs):
            milestone = Path(args.out).parent / f"encoder_epoch{epoch:04d}.pt"
            save_encoder(milestone, encoder, method="rotnet", args=args, epochs=epoch)
        save_checkpoint(last_path, {
            "encoder": encoder.state_dict(), "head": head.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch, "args": vars(args),
            "rng_state": capture_rng_state(device), "device": str(device),
            "torch_version": str(torch.__version__),
            "history": history,
        })
        completed_epoch = epoch

    if completed_epoch < args.epochs:
        print(f"Stopped at epoch {completed_epoch}/{args.epochs}; schedule is unchanged. "
              f"Resume from {last_path} to finish.")
        return
    save_encoder(args.out, encoder, method="rotnet", args=args, epochs=args.epochs)
    print(f"Saved encoder to {args.out}")


if __name__ == "__main__":
    main()
