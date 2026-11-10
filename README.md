# Self-Supervised Learning on CIFAR-10 (ECE 661, Team 38)

Comparing SimCLR, RotNet and SimSiam pretraining against a supervised baseline
on CIFAR-10, with a shared ResNet-20 encoder and fixed label subsets.

## Layout

| File | Purpose |
|---|---|
| `models.py` | ResNet-20 encoder (64-d features, optional per-stage outputs), `Classifier`, `load_encoder` |
| `data.py` | CIFAR-10 loaders and class-balanced label subsets (saved to `subsets/`) |
| `utils.py` | seeding, device selection, evaluation, checkpoint saving, `results/results.csv` logging |
| `train_classifier.py` | supervised baseline **and** fine-tuning of a pretrained encoder |
| `linear_eval.py` | frozen-encoder linear evaluation, incl. layer-wise (`--stage`) |
| `pretrain_rotnet.py` | RotNet pretraining |
| `pretrain_simclr.py`, `pretrain_simsiam.py` | *(to do)* |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CIFAR experiments use Python 3.10+ and the pinned PyTorch 2.10.0 / torchvision 0.25.0
pair, tested together on CPU. CUDA uses mixed precision; CPU and Apple MPS use float32.
GPU execution still needs verification on the machine used for the experiments.

CIFAR-10 downloads automatically into `data/` on first run.

## Commands

```bash
# Supervised baselines (random init) at 1%, 10%, 100% labels
python train_classifier.py --fraction 0.01
python train_classifier.py --fraction 0.1
python train_classifier.py --fraction 1.0

# RotNet pretraining
python pretrain_rotnet.py --epochs 100 --out checkpoints/rotnet.pt
# Resume with the SAME total epochs and other settings as the original command
python pretrain_rotnet.py --epochs 100 --out checkpoints/rotnet.pt --resume
# DCC GPU pilot (2 epochs; excludes known-bad GPU nodes). Run from /work/$USER/ece661-project4
mkdir -p slurm-logs && sbatch scripts/rotnet_pilot.sbatch

# Linear evaluation on frozen features
python linear_eval.py --checkpoint checkpoints/rotnet.pt
python linear_eval.py --checkpoint checkpoints/rotnet.pt --stage 2 --pool 2   # layer-wise
python linear_eval.py --random-init                                          # sanity floor

# Fine-tune a pretrained encoder on limited labels
python train_classifier.py --fraction 0.01 --init checkpoints/rotnet.pt
# Optional classifier save (parent folders are created automatically)
python train_classifier.py --fraction 0.01 --init checkpoints/rotnet.pt --save checkpoints/rotnet_1pct.pt
```

RotNet resumes from the last completed epoch in `<out>.last`, restoring model, optimizer,
scheduler, scaler, and Python/NumPy/PyTorch random states. A missing checkpoint or changed
settings produces an error instead of silently starting a different experiment. `--epochs`
is the original total, not an additional number of epochs. Old `.last` files without
settings and random states cannot resume reliably; their final encoder files still work
for evaluation. Exact reproducibility also depends on hardware, library versions, and
deterministic kernels; restoring random states alone cannot guarantee identical GPU results.

## Conventions (keep comparisons fair)

- **Every method uses `ResNet20` from `models.py`.** Pretraining scripts save the encoder
  with `utils.save_encoder(path, encoder, method=..., args=args)` so `linear_eval.py` and
  `train_classifier.py --init` work for every method.
- **Label subsets are generated in `subsets/` on first use. Commit and share these files.** Never regenerate them with a
  different seed for one method only.
- **Use the same downstream settings** (epochs, lr, batch size) for every method at a given
  label fraction.
- Every downstream evaluation run appends a row to `results/results.csv`, including
  subset seed, weight decay, optimizer, batch size, training steps per epoch, device,
  library versions, command settings, and the available source pretraining settings.
  Older CSV rows are preserved when adding the new columns; unknown old settings stay blank.
- At 1% labels, the 500 labeled images are sampled with replacement to request 5,000
  draws per training epoch (`--min-epoch-size 5000`). With batch size 128 and the last
  partial batch dropped, this is 39 steps / 4,992 draws. This reuses labels, not additional
  labeled images. Keep this setting identical across supervised and fine-tuning runs.
- Accuracy is reported on the CIFAR-10 test set. Don't tune hyperparameters on it; hold out
  part of the training set if you need to tune.
- Linear evaluation uses features extracted once without augmentation (faster than the
  SimCLR paper's protocol, which trains the linear layer with augmentation). Features are
  standardized using training statistics. Use this same adapted protocol, stage, and
  pooling size for comparisons between methods.

## Checks

```bash
python -m unittest discover -s tests -v
```

These CPU checks use small synthetic datasets, without downloading CIFAR-10. They cover
resume equivalence, rejected resume changes, saving, result metadata, and evaluation.
They do not establish CIFAR-10 convergence or final accuracy.

## References

1. Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations," ICML 2020. arXiv:2002.05709
2. Gidaris et al., "Unsupervised Representation Learning by Predicting Image Rotations," ICLR 2018. arXiv:1803.07728
3. Chen & He, "Exploring Simple Siamese Representation Learning," CVPR 2021. arXiv:2011.10566
