# Experiment plan

All methods use CIFAR-10 (50,000 training and 10,000 test images). Pretraining
uses the training images without class labels; downstream results are CIFAR-10
test top-1. The project proposal assigned SimCLR to Pengyu, RotNet to Teresa,
and SimSiam and cross-method analysis to both. [Proposal and references](references/README.md)

## Current plan · ResNet-50 (agreed 2026-11-21)

The team moved from ResNet-20 to the ResNet-50 that SimCLR Appendix B.9 uses for
CIFAR-10, so every method shares that encoder. The completed
[ResNet-20 study](#resnet-20-study--complete-for-rotnet) remains as a small-model comparison.

| Item | Setting |
|---|---|
| Backbone | ResNet-50 with the B.9 CIFAR stem: 3×3 stride-1 first conv, no max-pool, FC removed; 2048-d feature. Shared by RotNet, SimCLR, SimSiam, and the supervised baseline |
| Pretraining grid | Batch 256 and 1024 original images; 1,000 epochs; encoder saved every 100 epochs (10 per batch size) |
| Seed | 0 for pretraining, subsets, and every downstream run |
| RotNet pretraining | Four rotations per image, linear 4-way head. SGD with plain momentum 0.9 (RotNet paper), weight decay 5e-4, cosine, no warmup; LR 0.1 × batch / 256 (0.1 and 0.4); random crop and flip. Batch 1024 uses gradient accumulation, so batch norm sees sub-batches |
| SimCLR pretraining | B.9 recipe: LARS, base LR 1.5 × batch / 256, temperature 0.5, 10-epoch warmup, cosine, weight decay 1e-6 |
| Supervised baseline | Same ResNet-50 from random weights, trained with the fine-tune recipe below at batch 256 and 1024 on 1%, 3%, 5%, and 10% of the labels. Also trained end to end on 100% of the labels, as B.9 reports a supervised baseline "using the same architecture and batch size" |

### Downstream recipe

Every method and the supervised baseline use this recipe. SimCLR Section 6
fine-tunes "the whole base network" (Appendix B.5), and its linear evaluation
follows the same procedure with only the classifier trained (Appendix B.6).

| | Fine-tune (1%, 3%, 5%, 10%) | Linear evaluation (100%, linear only) |
|---|---|---|
| Starting point | Epoch-1,000 encoder at batch 256 and at batch 1024 | Every pretraining checkpoint (epochs 100–1,000) at both batch sizes, plus a random-init control |
| What trains | Whole network: encoder plus a new Linear(2048→10) | Linear(2048→10) only; the encoder stays frozen |
| Batch | Same as the encoder's pretraining batch (256 or 1024) | Same |
| Loss | Standard softmax cross-entropy over the 10 classes, no label smoothing | Same |
| Optimizer | SGD with Nesterov momentum 0.9 | Same |
| Starting LR | 0.05 × batch / 256: 0.05 and 0.2 (B.5) | 0.1 × batch / 256: 0.1 and 0.4 (B.6) |
| Weight decay | None | None |
| Augmentation | Random crop and horizontal flip | Random crop and flip, passed through the frozen encoder |
| Epoch length | At least 5,000 sampled images, drawn with replacement from small subsets | One pass over the 50,000 training images |
| Stopping | Constant LR. Multiply LR by 0.1 when training loss has not improved by more than 1% for 10 epochs. Stop at the second plateau or after 100 epochs | Same |
| Test top-1 | Measured on the final model after stopping. Curves may log test top-1 every 10 epochs, but test accuracy never decides when to stop | Same |
| RotNet runs | 2 encoders × 4 fractions = 8 | 20 checkpoints + 1 random init = 21 |
| Labels | Shared seed-0 class-balanced [subsets](../subsets/) | All 50,000 training labels |

None of the three papers states its downstream loss, so every downstream run uses
standard softmax cross-entropy. The stopping rule uses training loss only. A validation split would need labels
beyond the 1% budget, and test accuracy must stay untouched until reporting.
The 100-epoch cap exceeds the paper's fixed lengths (60 and 30 epochs for
fine-tuning, 90 for linear evaluation).

### Planned results table

Table 7 format, CIFAR-10 test top-1, seed 0. SimCLR Table B.4 is the top-1
version of Table 7.

| Method | Architecture | Batch | 1% | 3% | 5% | 10% | 100% (linear only) |
|---|---|---:|---:|---:|---:|---:|---:|
| Supervised baseline | ResNet-50 | 256 | | | | | — |
| Supervised baseline | ResNet-50 | 1024 | | | | | — |
| RotNet | ResNet-50 | 256 | | | | | |
| RotNet | ResNet-50 | 1024 | | | | | |
| SimCLR | ResNet-50 | 256 | | | | | |
| SimCLR | ResNet-50 | 1024 | | | | | |
| SimSiam | ResNet-50 | 256 | | | | | |
| SimSiam | ResNet-50 | 1024 | | | | | |

- **1%–10%:** the whole network is fine-tuned. For the supervised rows, the same network is trained from random weights.
- **100% (linear only):** frozen encoder plus a linear classifier trained on all labels, the epoch-1,000 point of Figure B.7.
- **Supervised rows at 100%:** there is no pretrained encoder to freeze, so the cell is empty. The supervised network trained end to end on all labels (B.9's reference, 95.1% in the paper) is reported beside the table.

| Owner | Runs |
|---|---|
| Teresa | 2 RotNet pretrains; 21 linear evaluations (20 checkpoints + random init); 8 RotNet fine-tunes |
| Pengyu | 2 SimCLR pretrains with probes and fine-tunes; 10 supervised runs (2 batches × 5 label settings) |
| Both | SimSiam at batch 256 and 1024, so its batch sensitivity can be compared with SimCLR's. Pengyu sets the SimSiam pretraining recipe; its downstream runs use the shared recipe above |

| Team total | Runs |
|---|---:|
| Pretraining: 3 methods × 2 batch sizes, 1,000 epochs | 6 |
| Linear evaluations: 6 runs × 10 checkpoints, plus a random-init control | 60 + 1 |
| Fine-tunes: 6 epoch-1,000 encoders × 4 label fractions | 24 |
| Supervised: 2 batch sizes × (1%, 3%, 5%, 10%, 100%) | 10 |

## ResNet-20 study · complete for RotNet

| Method | What the encoder learns from | Part 1 · Frozen probes | Part 2 · Train the full classifier |
|---|---|---|---|
| RotNet | Predict which of four rotations was applied | 15 runs × 10 checkpoints, complete | Fine-tune the designated encoder at each label fraction, complete |
| SimCLR | Bring two views of the same image together and separate views of other images | Superseded by the ResNet-50 plan | Superseded |
| SimSiam | Predict the other view's representation with a stop-gradient target | Superseded | Superseded |
| Supervised | Learn directly from class labels | — | Complete alongside RotNet |

### Part 1 · Batch size and pretraining duration

A **frozen probe** keeps the encoder fixed and trains a fresh linear classifier
on its features. It measures how easily those features separate the ten classes.

| Setting | RotNet | SimCLR (planned) |
|---|---|---|
| Batch sizes in original images | 256, 512, 1024, 2048, 4096 | Same |
| Pretraining learning rates | 0.03, 0.1, 0.3 with no batch scaling | Base rates 0.5, 1.0, 1.5 × batch size / 256 |
| Views per image | Four rotations | Two independently augmented views |
| Schedule and snapshots | 1,000 epochs, save every 100 | Same |
| Grid and probe inputs | [Runs](../experiments/rotnet/sweep.csv) · [Probes](../experiments/rotnet/milestones.json) | [Runs](../experiments/simclr/sweep.csv) · [Probes](../experiments/simclr/milestones.json) |

The project adapts SimCLR Figure B.7 to ResNet-20. Each short-duration point is
a snapshot of a 1,000-epoch schedule. For each batch and epoch, plot the mean
and sample standard deviation of accuracy **across the three learning rates**.
All runs use seed 0. Record GPU time and optimizer steps alongside epoch counts
because the methods process different numbers of views.

| Main-table encoder | Prespecified setting |
|---|---|
| RotNet | Batch 1024, LR 0.1, epoch 1000 |
| SimCLR | Batch 1024, base LR 1.0 (peak 4.0), temperature 0.5, epoch 1000, subject to its pilot |
| SimSiam | One method-specific recipe at the same feasible batch and epoch setting |

Reuse the individual anchor probes in the main table. Use a prespecified setting
or a training-only validation split to select checkpoints and learning rates.
SimSiam's single run supplies a main-table comparison. Batch-size trends come
from the RotNet and SimCLR grids.

### Part 2 · Label efficiency

For each method and label fraction, start a separate fine-tune from that
method's designated encoder. Run one supervised baseline per fraction with the
same subsets and classifier settings. RotNet also fits one frozen probe per
fraction to measure the contribution of fine-tuning. SimCLR Table 7 uses 1% and
10%. We add 5%, 20%, and 100%, plus 8% (4,000 labels), the common CIFAR-10
semi-supervised setting. Report top-1. `results.csv` also records top-3.

| Label fraction | Labeled images | Images per class | Classifier updates per epoch |
|---:|---:|---:|---:|
| 1% | 500 | 50 | 39 |
| 5% | 2,500 | 250 | 39 |
| 8% | 4,000 | 400 | 39 |
| 10% | 5,000 | 500 | 39 |
| 20% | 10,000 | 1,000 | 78 |
| 100% | 50,000 | 5,000 | 390 |

Use the [saved seed-0 subsets](../subsets/) across methods. Below 5,000 images,
each epoch requests 5,000 draws with replacement from the subset. With batch
size 128 and the last partial batch dropped, 4,992 draws reach the optimizer.

| Setting | Frozen linear probe | Fine-tune or supervised baseline |
|---|---|---|
| Trainable weights | Linear classifier | Encoder and classifier |
| Input | Stage 3, pool 1, 64 features. Extract once in eval mode without augmentation and standardize using training statistics. | CIFAR-10 images with random crop and horizontal flip |
| Optimizer | Adam, LR 0.001, weight decay 0.00001 | SGD, LR 0.1, momentum 0.9, weight decay 0.0005 |
| Schedule | 100 epochs, cosine decay, batch 256 | 100 epochs, cosine decay, batch 128 |
| Seeds | Training 0, subset 0 | Training 0, subset 0 |

## Remaining implementation and report

| Work | What to check |
|---|---|
| SimCLR trainer | Two CIFAR views with crop, flip and color distortion (strength 0.5), no blur. Check the projection head, normalized NT-Xent, self-pair masking and the full set of negatives. Document LARS, warmup/cosine, parameter exclusions and weight decay. Small contrastive batches combined by gradient accumulation still use smaller negative pools. |
| SimSiam trainer | Check symmetric negative cosine loss, stopped targets, projector/predictor batch normalization, encoder exports, and feature spread to detect collapse. Record the SGD recipe and predictor schedule. |
| GPU pilots | Check finite loss, peak memory, epoch time, and checkpoint restart at the planned batch and schedule. For sweeps, verify the largest physical batch, all rates, and the measured walltime budget before releasing full training. |
| Report | Include both Figure B.7-style plots, the single-seed anchor table, the accuracy-by-label-fraction comparison, and measured compute. Document missing runs and any scope change. |

The plan targeted Part 2 by Nov 25 and an experiment freeze by Dec 2. The
course guidelines give the Dec 9, 1–4 pm
presentation slot. The [DCC guide](slurm.md) covers job setup and restart.
