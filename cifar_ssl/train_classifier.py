"""Train ResNet-20 + linear layer on a fraction of CIFAR-10 labels.

Two uses, same code so the comparison is fair:
  supervised baseline:  python -m cifar_ssl.train_classifier --fraction 0.01
  fine-tune pretrained: python -m cifar_ssl.train_classifier --fraction 0.01 --init checkpoints/rotnet.pt
"""
import argparse

import torch
import torch.nn as nn

from .data import labeled_train_loader, test_loader
from .models import Classifier, ResNet20, load_encoder
from .utils import autocast_context, evaluate, evaluate_topk, get_device, log_result, save_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fraction", type=float, default=1.0, help="fraction of labels: 0.01, 0.1, 1.0")
    p.add_argument("--init", default=None, help="pretrained encoder checkpoint (omit for random init)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subset-seed", type=int, default=0, help="which saved label subset to use")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--min-epoch-size", type=int, default=5000,
                   help="minimum training draws per epoch; small subsets sample with replacement")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--save", default=None, help="optional path to save the trained classifier")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    use_amp = device.type == "cuda"

    if args.init:
        encoder, ckpt = load_encoder(args.init)
        method = ckpt["method"]
        experiment = "finetune"
    else:
        ckpt = {}
        encoder = ResNet20()
        method = "supervised"
        experiment = "supervised"
    model = Classifier(encoder).to(device)

    train_loader = labeled_train_loader(args.fraction, args.batch_size, seed=args.subset_seed,
                                        num_workers=args.num_workers,
                                        min_epoch_size=args.min_epoch_size)
    eval_loader = test_loader(num_workers=args.num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"{experiment} | method={method} | fraction={args.fraction} | device={device} | "
          f"{len(train_loader)} steps/epoch")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = correct = seen = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast_context(device):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * targets.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            seen += targets.size(0)
        scheduler.step()

        msg = f"epoch {epoch:3d} | loss {total_loss / seen:.4f} | train acc {correct / seen:.4f}"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            msg += f" | test acc {evaluate(model, eval_loader, device):.4f}"
        print(msg)

    test = evaluate_topk(model, eval_loader, device, ks=(1, 3))
    test_acc = test[1]
    print(f"final | test top-1 {test[1]:.4f} | test top-3 {test[3]:.4f}")
    if args.save:
        save_checkpoint(args.save, {"model": model.state_dict(), "args": vars(args),
                                    "pretrain_args": ckpt.get("args", {})})
    log_result(method=method, experiment=experiment, label_fraction=args.fraction,
               test_acc=round(test_acc, 4), test_top3=round(test[3], 4),
               epochs=args.epochs, batch_size=args.batch_size,
               lr=args.lr, seed=args.seed, checkpoint=args.init or "",
               subset_seed=args.subset_seed, weight_decay=args.weight_decay,
               optimizer="SGD", steps_per_epoch=len(train_loader),
               min_epoch_size=args.min_epoch_size, device=str(device), config=vars(args),
               pretrain_config=ckpt.get("args", {}))


if __name__ == "__main__":
    main()
