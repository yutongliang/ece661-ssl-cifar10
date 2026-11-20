# Experiment plan

All methods use the same CIFAR-10 split and [ResNet-20 encoder](../cifar_ssl/models.py).
Pretraining uses all 50,000 training images without class labels. The downstream
experiments measure CIFAR-10 class accuracy on the 10,000 test images.

| Method | What the encoder learns from | Part 1 · Frozen probes | Part 2 · Train the full classifier |
|---|---|---|---|
| RotNet | Predict which of four rotations was applied | 15 runs × 10 checkpoints, complete | Fine-tune the designated encoder at each label fraction |
| SimCLR | Bring two views of the same image together and separate views of other images | 15 runs × 10 checkpoints, planned | Fine-tune the designated encoder at each label fraction |
| SimSiam | Predict the other view's representation with a stop-gradient target | One encoder and one probe for the main table, planned | Fine-tune that same encoder at each label fraction |
| Supervised | Learn directly from class labels | — | Train from scratch at each label fraction |

The project proposal assigned SimCLR to Pengyu, RotNet to Teresa, and SimSiam
and cross-method analysis to both. [Proposal and references](references/README.md)

## Part 1 · Batch size and pretraining duration

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

## Part 2 · Label efficiency

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
