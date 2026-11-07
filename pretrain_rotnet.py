"""RotNet pretraining (Gidaris et al., ICLR 2018) on CIFAR-10, no class labels.

Every image is shown at 0/90/180/270 degrees and the network predicts which
rotation was applied (4-way classification). After training, only the encoder
is kept; the rotation head is thrown away.

    python pretrain_rotnet.py --epochs 100
    python pretrain_rotnet.py --epochs 100 --resume   # continue after a disconnect
"""
import argparse
import os

import torch
import torch.nn as nn

from data import cifar10, test_transform, train_transform
from models import ResNet20
from utils import (autocast_context, capture_rng_state, get_device, restore_rng_state,
                   save_checkpoint, save_encoder, set_seed)

NUM_ROTATIONS = 4


def load_resume_state(args, device):
    """Resume the same planned run, never silently change its training schedule."""
    path = args.out + ".last"
    if not os.path.isfile(path):
        raise ValueError(f"Cannot resume: checkpoint {path} does not exist")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not {"args", "rng_state", "device", "torch_version"}.issubset(state):
        raise ValueError("Legacy resume checkpoint lacks settings/RNG state. Start a new run "
                         "with a new --out path; existing final encoders remain usable.")
    differences = [f"{key}: saved={state['args'].get(key)!r}, requested={value!r}"
                   for key, value in vars(args).items()
                   if key not in {"out", "resume"} and state["args"].get(key) != value]
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128, help="images per batch (x4 after rotation)")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--out", default="checkpoints/rotnet.pt", help="final encoder checkpoint")
    p.add_argument("--resume", action="store_true", help="resume from <out>.last")
    return p.parse_args()


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
    use_amp = device.type == "cuda"
    # Validate before downloading data or constructing a new training run.
    state = load_resume_state(args, device) if args.resume else None

    # Class labels come with the dataset but are never used for training.
    train_loader = torch.utils.data.DataLoader(
        cifar10(train=True, transform=train_transform), batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        cifar10(train=False, transform=test_transform), batch_size=256,
        shuffle=False, num_workers=args.num_workers)

    encoder = ResNet20().to(device)
    head = nn.Linear(encoder.feature_dim, NUM_ROTATIONS).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss()

    # Full training state, saved every epoch so a lost session can resume.
    last_path = args.out + ".last"
    start_epoch = 1
    if state is not None:
        encoder.load_state_dict(state["encoder"])
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        # Restore after model initialization, before creating any loader iterator.
        restore_rng_state(state["rng_state"], device)
        start_epoch = state["epoch"] + 1
        print(f"Resumed from {last_path} at epoch {start_epoch}")

    print(f"RotNet pretraining | device={device} | {len(train_loader)} steps/epoch")
    for epoch in range(start_epoch, args.epochs + 1):
        encoder.train()
        head.train()
        total_loss = correct = seen = 0
        for images, _ in train_loader:
            images, labels = rotate_batch(images)
            images, labels = images.to(device), labels.to(device)
            with autocast_context(device):
                logits = head(encoder(images))
                loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)
        scheduler.step()

        msg = f"epoch {epoch:3d} | loss {total_loss / seen:.4f} | rotation acc {correct / seen:.4f}"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            msg += f" | test rotation acc {rotation_accuracy(encoder, head, test_loader, device):.4f}"
        print(msg)

        save_checkpoint(last_path, {
            "encoder": encoder.state_dict(), "head": head.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch, "args": vars(args),
            "rng_state": capture_rng_state(device), "device": str(device),
            "torch_version": str(torch.__version__),
        })

    save_encoder(args.out, encoder, method="rotnet", args=args, epochs=args.epochs)
    print(f"Saved encoder to {args.out}")


if __name__ == "__main__":
    main()
