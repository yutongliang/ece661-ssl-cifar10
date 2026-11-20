# Self-supervised learning on CIFAR-10

ECE 661, Team 38. We compare how well RotNet, SimCLR, and SimSiam learn image
features with a shared ResNet-20, then measure how much those features help when
few class labels are available.

| Work | Status |
|---|---|
| Part 1 · Batch size and pretraining duration | RotNet complete with 15 pretraining runs and 150 frozen probes. The designated RotNet checkpoint scored **65.97%** CIFAR-10 top-1. SimCLR is planned. |
| Part 2 · Learning with 1%–100% of the labels | RotNet fine-tunes, supervised baselines, and frozen probes at 1, 5, 8, 10, 20, and 100% are running on DCC. SimCLR and SimSiam trainers are still needed in this checkout. |

## Start here

| I want to… | Go to |
|---|---|
| See the RotNet plot, scores, or original logs | [Results](results/rotnet/README.md) |
| Understand what we compare and why | [Experiment plan](docs/experiments.md) |
| Submit or monitor a GPU job | [DCC guide](docs/slurm.md) |
| Find the papers or course documents | [References](docs/references/README.md) |

## Run locally

From the repository root, use Python 3.10 or newer in an environment that can
install the pinned dependencies. CIFAR-10 downloads to `data/` on first use.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

| Task | Example command |
|---|---|
| Train RotNet | `python -m cifar_ssl.pretrain_rotnet --epochs 100 --out checkpoints/rotnet.pt` |
| Resume that run | `python -m cifar_ssl.pretrain_rotnet --epochs 100 --out checkpoints/rotnet.pt --resume` |
| Fit a classifier on frozen RotNet features | `python -m cifar_ssl.linear_eval --checkpoint checkpoints/rotnet.pt` |
| Measure the random-feature control | `python -m cifar_ssl.linear_eval --random-init` |
| Fine-tune RotNet using 1% of the labels | `python -m cifar_ssl.train_classifier --fraction 0.01 --init checkpoints/rotnet.pt` |
| Train from scratch using the same 1% | `python -m cifar_ssl.train_classifier --fraction 0.01` |
| Run CPU tests on synthetic data | `python -m unittest discover -s tests -v` |

Use `--fraction` with 0.01, 0.05, 0.08, 0.1, 0.2, or 1.0. Each value except
1.0 has a saved subset. Downstream runs append to `results/results.csv`. Fine-tuning and
supervised training can save a classifier with `--save PATH`. For a probe on
an intermediate layer, pass `--stage 2 --pool 2` to `cifar_ssl.linear_eval`.

## Find the code and inputs

| Path | Contents |
|---|---|
| [cifar_ssl/](cifar_ssl/) | Training, frozen evaluation, CIFAR-10 loading, and ResNet-20 |
| [experiments/](experiments/) | RotNet and SimCLR run grids and lists of checkpoints to probe |
| [subsets/](subsets/) | Shared class-balanced index files for the 1%, 5%, 8%, 10%, and 20% splits |
| [scripts/](scripts/) | Plotting and paper downloads. Cluster scripts live in [scripts/slurm/](scripts/slurm/). |
| [tests/](tests/) | Synthetic training, checkpoint, and Slurm submission checks |
