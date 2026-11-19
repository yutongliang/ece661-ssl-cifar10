# Team 38 execution plan — 2026-11-15

Status: proposal, course requirements and relevant paper sections reviewed; all
five PDFs downloaded and parsed successfully. After the user successfully ran
`git fetch --all --prune`, the fetched origin still contains only `rotnet` at
`5e5986b` (November 10), with no new commits. No SimCLR/SimSiam branch or teammate
changes were found in those fetched refs. The user subsequently confirmed there
was no new push, so no teammate update remains to be fetched for this plan.
SimCLR and SimSiam still need implementation in this checkout. The user successfully
submitted the RotNet pipeline on 2026-11-15: preflight `61863269`, pilot `61863270`,
15-model array `61863271`. The supplied submission output shows preflight running
and both GPU stages pending on their expected dependencies. These are submission
observations, not proof of completed GPU pilots or model training. The launch ID
is `20261115T230242708810Z`; 290 pending ECE662 jobs were confirmed deprioritized.
Immediate order: RotNet's 15-model pretraining grid first, then its frozen linear
evaluations; fine-tuning and supervised baselines are deferred to Part 2.
The launch package is ready; see [RotNet submission instructions](scripts/ROTNET_SWEEP.md).

Latest progress (direct SSH/sacct, 2026-11-18): **RotNet pretraining is complete.**
All 15 tasks of `61863271` finished COMPLETED, exit 0:0, each with a verified
`complete.json`; outputs are copied to `dcc_results/rotnet_sweep_20261115/`.
Logged accuracies are rotation (pretext) top-1 only; final test rotation accuracy
ranges 0.881 (B=4096, lr 0.03) to 0.932 (B=256, lr 0.1). These do not select the
main-table encoder, which remains the prespecified B=1024, lr=0.1 anchor.
The 150 frozen linear probes plus a random-init control were submitted as
`62101838` (`scripts/rotnet_linear_probes.sbatch`); on 2026-11-18 all 16 tasks were
PENDING (Priority) in gpu-common. See [ROTNET_SWEEP.md](scripts/ROTNET_SWEEP.md).
SimCLR and SimSiam trainers still do not exist in this checkout.

## 1. What we have committed to

Sources: `../661 Project Proposal.pdf`, pages 1–3;
`../ECE661 Project Topic List-26FA (1).pdf`, page 5;
`../ECE661 Project Guidelines-26FA.pdf`, pages 1–2.

| Deliverable | Proposal / course requirement | Execution decision |
|---|---|---|
| Shared architecture | Proposal: ResNet-20 for all methods | Retain `models.ResNet20`; 64-d final features |
| Methods | SimCLR and RotNet required; SimSiam is the chosen addition | Finish all three before expanding unrelated experiments |
| Pretraining data | All 50,000 CIFAR-10 training images, no class labels | Never use test labels to train SSL encoders |
| Representation quality | Frozen-feature linear evaluation | Same primary probe protocol and feature stage for every method |
| Figure B.7 | Proposal: reduced grid and trends; course: feasible settings | User-selected Part 1: 15 SimCLR + 15 RotNet trajectories |
| Label efficiency | Class-balanced 1%, 10%, 100%; compare with training from scratch | Same saved subsets and downstream settings for each fraction |
| Additional method | SimSiam in the proposal | One encoder for the main linear-evaluation and fine-tuning comparisons; no SimSiam sweep |
| Ownership | Pengyu: SimCLR; Teresa: RotNet; jointly: SimSiam and comparison | Inspect existing teammate work before duplicating implementation |

The submitted proposal explicitly anticipates a smaller model, limited compute,
and reporting trends rather than exact paper accuracies. A separate ResNet-50
replication would be an extension, not a prerequisite for this plan. Keep the
requested five batch sizes and three learning rates if memory and runtime allow.

## 2. Current evidence

- `checkpoints/rotnet.pt` and `.last` are copied and byte-verified against the
  supplied Downloads/results files. Job 61832176 completed 100 epochs on a 2080 Ti.
- Final training rotation accuracy was 96.44%; test rotation accuracy was 92.45%.
  These are pretext-task metrics, not CIFAR-10 class accuracies.
- Static inspection of checkpoint metadata confirms epochs=100, batch_size=128,
  lr=0.1, weight_decay=0.0005, seed=0, num_workers=4, eval_every=1.
  The resume state records device=cuda and PyTorch `2.10.0+cu126`.
  Loading the tensors into the model still needs the DCC environment.
- The prior full-run log reports eight passing synthetic tests. A matching local
  environment was subsequently found at `/opt/anaconda3/envs/persona310/bin/python`
  (torch 2.10.0 / torchvision 0.25.0); the submitted bundle's 13-test suite passes there.
  These tests cover planned-stop/resume equivalence, actual milestone weights,
  non-finite-loss/CUDA guards, grid mapping, source snapshots and priority filters.
- A first submission attempt failed on `flock` contention; retry succeeded. The
  precise cause is unconfirmed. Local lock hardening now uses atomic directory
  creation with owner metadata and conservative stale-owner recovery, with 16
  tests passing. This later change has not been applied to the active submission's
  frozen source, which remains untouched. Do not resubmit just to deploy it.
- No downstream results table or SimCLR/SimSiam implementation exists in the
  local branch or the fetched origin refs. The user confirmed no new push; plan
  implementation from this verified baseline rather than waiting for new commits.
- All five research PDFs passed signature/checksum checks and were opened with
  PDFKit: SimCLR 20 pages, RotNet 16, SimSiam 10, ResNet 12 and LARS 8.
- `scripts/rotnet_downstream.sbatch` is prepared and shell-checked. It contains
  eight seed-0 tasks: two full-label frozen probes, three supervised baselines,
  and three RotNet fine-tunes. It is ready for an environment/checkpoint smoke
  check but is deferred: the accepted immediate task is RotNet pretraining first.

## 3. Accepted scope: two parts

All methods use the shared ResNet-20. The first pass uses training seed 0 and
subset seed 0. No additional seed repetitions are scheduled in this scope.

The main results compare all three methods under a selected common batch size
and epoch horizon, using both frozen linear evaluation and limited-label
fine-tuning. The full batch-size/epoch grid is an additional comparison between
SimCLR and RotNet. SimSiam participates in the main results without a full grid.

| Method | Full Part 1 grid | Main frozen linear evaluation | Part 2: 1%, 10%, 100% labels |
|---|---|---|---|
| SimCLR | 15 models | One designated grid encoder; reuse its final milestone probe | 3 independent fine-tunes from that encoder |
| RotNet | 15 models | One designated grid encoder; reuse its final milestone probe | 3 independent fine-tunes from that encoder |
| SimSiam | No grid | Train one encoder and fit one matching linear probe | 3 independent fine-tunes from the same encoder |
| Supervised baseline | No SSL pretraining | Not applicable | 3 runs from random initialization |

The batch-size grid is not needed to include a method in the main comparison.
Conversely, a single SimSiam batch size cannot establish that it is less sensitive
to batch size or less dependent on large batches. The proposal raised that
additional question; testing it is deferred, and the report must identify this
scope adjustment rather than claim that the current runs answer it. SimSiam's
lack of negative pairs alone does not establish batch-size insensitivity.

### Part 1: batch size / training duration, with frozen linear evaluation

| Method | Pretraining grid | Models | Figure / comparison |
|---|---|---:|---|
| SimCLR | B={256,512,1024,2048,4096} × 3 learning rates; 1,000 epochs each | 15 | Adapt SimCLR Figure B.7 |
| RotNet | Same five batches × 3 method-appropriate learning rates; 1,000 epochs each | 15 | Same Figure B.7-style evaluation, using RotNet features |
| SimSiam | No batch-size grid | 0 grid models | Included in the main comparison through one encoder and one linear probe |

For SimCLR and RotNet, save encoder snapshots at epochs 100,200,…,1000 and fit a
fresh linear classifier on each frozen snapshot. Each of these methods has 150
milestone evaluations, but
these are small classifier fits on the 15 trajectories, not 150 SSL pretraining
runs. Evaluate CIFAR-10 category accuracy, rather than rotation-task accuracy.
Aggregate rates into a mean and SD at each batch/epoch. The three runs vary
learning rate, not random seed; their SD must be labeled accordingly.

SimCLR's planned base rates are {0.5,1.0,1.5}, temperature=0.5, with explicit
linear scaling `peak_lr = base_lr * B / 256`. The
[authors' scheduler](https://github.com/google-research/simclr/blob/master/tf2/model.py)
provides this convention. RotNet uses prespecified SGD rates {0.03,0.1,0.3}
without batch scaling, centered on the earlier successful 0.1 recipe. All 15
configurations must pass five-epoch GPU pilots before full training is released;
these settings are not a claim of optimal rates. The pilot and full run preserve
the same 1000-epoch scheduler horizon and resume the same trajectory.

Manifests: [SimCLR grid](experiments/figure_b7_sweep.csv),
[SimCLR milestones](experiments/figure_b7_milestones.json),
[RotNet grid](experiments/rotnet_figure_b7_sweep.csv),
[RotNet milestones](experiments/rotnet_figure_b7_milestones.json).

This adapts the [paper's Figure B.7](papers/simclr-chen2020.pdf), which uses
ResNet-50, to our shared ResNet-20. A 100-epoch snapshot from a 1,000-epoch cosine
schedule differs from a completed 100-epoch schedule; call our plot a checkpoint
trajectory experiment rather than exact schedule reproduction. We are not
scheduling separate training for every duration.

The completed B=128/100-epoch RotNet remains useful for plumbing checks, but it
is outside the selected five-batch/1,000-epoch grid. Preserve it and start new
outputs; do not change its original resume horizon.

### Part 2: limited labels, with fine-tuning

Select ONE pretrained encoder per method, then start a separate classifier run
from that same encoder for each label fraction. Do not sequentially fine-tune
1% then 10% then 100%, or fine-tune every model from Part 1.

| Method | Encoder source | 1% labels | 10% labels | 100% labels | Classifier fits | Figure / comparison |
|---|---|---|---|---|---:|---|
| SimCLR | One designated Part 1 model | Fine-tune | Fine-tune | Fine-tune | 3 | SimCLR Section 6 / Table 7 adapted to CIFAR-10 |
| RotNet | One designated Part 1 model | Fine-tune | Fine-tune | Fine-tune | 3 | Same limited-label comparison; RotNet Figure 5(b) is supporting context |
| SimSiam | Reuse the single encoder from its main linear comparison | Fine-tune | Fine-tune | Fine-tune | 3 | Project extension applying the same limited-label protocol |
| Supervised baseline | Random initialization, no SSL pretraining | Train from scratch | Train from scratch | Train from scratch | 3 | Reference for the three methods above |

Part 2 therefore needs 9 fine-tunes plus 3 baseline fits. The single SimSiam
pretraining run supplies both its linear probe and all three fine-tunes; do not
train a separate SimSiam encoder for each evaluation. Reuse the same saved
class-balanced subsets and downstream
settings across all four rows. Choose the designated encoders before examining
final test scores, or select using a training-only validation protocol.

Working anchor: B=1024 and epoch=1000 for all three pretrained encoders. SimCLR
uses base LR=1.0; RotNet uses its prespecified stable middle rate. SimSiam uses
one separately validated method-appropriate recipe. If the anchor cannot fit,
record the revised common setting before evaluating its test accuracy.

Use these same three designated encoders for the main frozen-linear table. Reuse
the individual final milestone scores for the selected SimCLR/RotNet runs and
fit one identical probe on SimSiam. Do not compare their three-rate grid averages
against SimSiam's single-run score as if these were the same statistical quantity.
The main table reports three single-seed anchor results, with no seed-error bars.

Current long-run scope: 30 Part 1 SSL trajectories, one SimSiam encoder shared by
the main comparisons, and 12 Part 2 classifier fits. Frozen linear evaluation
comprises 300 grid milestone fits plus one SimSiam anchor fit. The main SimCLR
and RotNet anchor scores are already included in those 300 fits.
The existing 100-epoch RotNet is a completed preliminary run, not one of these
30 new grid trajectories. Short pilots are verification steps, not extra research
ablations. SimSiam batch-size sweeps, layer-wise studies, random-feature controls,
and additional seeds are deferred.

## 4. Fixed protocols before long runs

### Data and downstream comparison

Use the existing 500-image (1%) and 5,000-image (10%) class-balanced JSON subsets;
generate them once before launching concurrent jobs, then share and checksum them.
The full-label set has 50,000 images. The existing code saves subsets relative to
the source checkout, so separate checkouts need the same files copied in.

Retain the current downstream recipe for the first comparison: 100 epochs,
batch=128, SGD momentum=0.9, lr=0.1, weight decay=5e-4, cosine decay, seed 0,
subset seed 0, min epoch size 5,000, eval every 10 epochs. These are project
defaults, not claimed optimal fine-tuning hyperparameters. With dropped partial
batches, the 1%/10% cases each have 39 updates per epoch; 100% has 390.

If a revised downstream recipe is selected using validation data, rerun the
affected methods under that same recipe rather than tuning only one method on the
test set. Keep test accuracy for final reporting. Select a designated checkpoint
or hyperparameters using a training-only validation split or a prespecified rule;
do not select the best Figure B.7 test bar for the main comparison.
The prespecified SimCLR anchor is B=1024, base_lr=1.0, temperature=0.5, seed=0,
epoch=1000, contingent on its pilot passing. If it cannot fit, record an amended
anchor before seeing its test score and use the feasible common batch for SimSiam.

### Frozen-feature evaluation

Primary feature definition: stage 3, pool=1, 64 dimensions, 100% training labels.
Use the same `linear_eval.py` protocol for every method: encoder in evaluation
mode, no encoder or BN updates, features extracted without augmentation,
standardized using training statistics, then a linear classifier. Existing defaults
are Adam, lr=1e-3, weight_decay=1e-5, batch=256, 100 epochs.

This is the repo's fast adapted protocol. The [authors' linear-evaluation code](https://github.com/google-research/simclr)
uses a different training recipe. Keep the current probe for within-project
comparisons and the 300 milestone-probe budget across SimCLR and RotNet; label the deviation in the report. A
paper-style augmented probe is a separate optional experiment, not a mix of rows
in one figure. Cache features by encoder checksum, feature stage/pool and transform
to avoid unnecessary extractions when repeating only classifier fits. Apply this
same feature definition and probe recipe to the one SimSiam anchor fit.

### SimCLR implementation audit

Before the sweep, verify two independently augmented views, normalized embeddings,
a nonlinear projection head, correct positive indices and exclusion of self-pairs
in NT-Xent. Check the full global negative set if distributed training is used.
The CIFAR recipe calls for random resized crops/flips, color distortion strength
0.5 and no blur; use temperature 0.5. For paper-inspired optimization, explicitly
document LARS, warmup/cosine scheduling, parameter exclusions and regularization;
the paper's default weight decay is 1e-6.

Do not silently substitute the official README's separate ResNet-18 CIFAR recipe
(which uses weight decay 1e-4), or the existing RotNet crop-and-flip-only transform.
The [official README](https://github.com/google-research/simclr/blob/master/README.md)
is useful implementation guidance but does not make its demonstration command
identical to Figure B.7.

Global batch B means B original images and 2B augmented views. At B=4096,
the naive similarity matrix has 8192² entries: 256 MiB in float32, excluding
gradients, encoder activations and other allocations. Plain gradient accumulation
over smaller contrastive batches does not supply the same negatives. Prefer a
verified larger-memory single GPU; use distributed gathering/synchronized BN only
after correctness checks. Do not call accumulation alone a reproduced B=4096 run.

### SimSiam implementation and comparison

Audit symmetric negative-cosine loss with stopped targets, predictor behavior,
projector/predictor BN, and encoder-only exports. The [paper's CIFAR appendix](https://arxiv.org/html/2011.10566v1#A4)
uses ResNet-18, a two-layer 2048-d projector, batch 512 and 800 epochs with SGD,
base lr=0.03, momentum=0.9, weight_decay=5e-4 and no blur. Our ResNet-20 and the
comparison below are documented adaptations.

Proposed anchor: same B=1024 and 1,000-epoch horizon as the SimCLR anchor, using
the same two-view augmentations. Start from the CIFAR SGD recipe, explicitly
record scaling and predictor schedule, and validate numerics before committing
to the full run. Do not reuse SimCLR's LARS rates unchanged. Log normalized feature
standard deviations on a fixed sample and a collapse diagnostic throughout.

Train only this one SimSiam encoder. Include its final frozen linear probe in
the main three-method table, then reuse the encoder for Part 2 fine-tuning.
Its batch-size ablation is outside the current scope. Feature-spread logging is
a training diagnostic, not a new grid experiment. Do not claim that this
single-batch run establishes lower batch-size dependence.

### RotNet reuse

Use the completed 100-epoch encoder for smoke checks. Its
[source paper](papers/rotnet-gidaris2018.pdf) uses NIN for CIFAR; our ResNet-20,
cosine schedule and probe are adaptations. The user-selected Part 1 now needs
15 new RotNet trajectories with a 1,000-epoch horizon and milestone exports.
Do not resume the finished 100-epoch run with `--epochs 1000`; start new paths
with the original total horizon fixed at 1000 from initialization.

Use the same original-image batch sizes and epoch horizon for Part 1, while
recording views/image, optimizer updates and measured GPU time. RotNet uses four
rotations per image and SimCLR uses two augmented views, so matching epochs does
not establish equal compute. Report that distinction without adding another
budget-matching ablation to the current scope.

## 5. DCC launch sequence

### Stage A: record the environment and validate artifacts

Run on the DCC login node for inspection and scheduling; run training inside a
Slurm allocation. Do not reinstall/upgrade the successful environment casually.

```bash
cd "/work/$USER/ece661-project4"
git status --short
git rev-parse HEAD
git branch -a
sinfo -p gpu-common -o '%P %l %a %G'
sinfo -N -p gpu-common -o '%N %G %t'
scontrol show partition gpu-common
squeue -u "$USER"
```

In the activated GPU environment, record Python/PyTorch/torchvision/CUDA versions,
GPU model, code SHA plus dirty diff, and a full environment export alongside each
run. Load `rotnet.pt` with `models.load_encoder`, do one feature/probe batch, and
run `python -m unittest discover -s tests -v` before the initial full jobs. Save
checkpoint checksums. Do not treat a successful static file copy as a model-load test.

Current [DCC hardware documentation](https://oit-rc.pages.oit.duke.edu/rcsupportdocs/dcc/architecture/)
lists 2080 Ti (11 GB), A5000 (24 GB) and RTX 5000 Ada (32 GB) in gpu-common;
confirm live GRES names and account access. Use the successful 2080 path for
RotNet/downstream work. Pilot SimCLR's largest batches on A5000/Ada if needed.
H200 access requires an allocation; it is not assumed in this plan
([access policy](https://oit-rc.pages.oit.duke.edu/rcsupportdocs/dcc/h200/)).

### Stage B: submit RotNet pretraining first

Run on the Mac:

```bash
bash /Users/teresaliang/Desktop/ECE661/project4/scripts/launch_rotnet_dcc.sh
```

This uploads the prepared bundle into the separate DCC directory
`/work/yl1064/ece661-rotnet-sweep`, then submits CPU preflight -> five-epoch pilots
for all 15 configurations -> the 15-task full-training array, using `afterok`
dependencies. It does not submit baseline, fine-tuning or linear-evaluation jobs.
The latter uses the saved milestones after training completes.

The initial submission requested `scavenger-gpu`. The user's later live inventory
shows three gpu-common nodes with eight `5000_ada` GPUs each, versus a smaller
scavenger pool of that type. Future submissions now default to `gpu-common`, one
32 GB `5000_ada` per task, with at most two full-training tasks running at once.
The user confirmed an in-place partition change for jobs 61863270 and 61863271:
both now show gpu-common, state PD and reason None. They have not started in
that observation. Job IDs are unchanged and no new submission was made.
Mixed node state does not establish GPU availability. The four ECE662 exclusion
nodes are included. Confirmed GPU fit and timing still come from the pilot;
failed pilots block full training instead of changing requested batches silently.
Each epoch saves full resumable state; every 100 epochs exports an encoder.

Source files/configuration hashes are frozen per submission. The launcher checks
room for 17 queued jobs under the 400-job limit and refuses duplicate active
campaigns. It lowers only currently pending, owned `cifar10_lrms3...` jobs' Nice
priority and records previous values for restoration. Running ECE662 jobs and
its `lrm_seeds` supervisor continue. This does not guarantee cross-partition
ordering or reclaim occupied GPUs; new supervisor submissions require a fresh
queue review. See [submission and restore instructions](scripts/ROTNET_SWEEP.md).

The older eight-task downstream script and the separate supervised-baseline
script remain available but are not part of the current submission.

### Stage C: SimCLR, RotNet and SimSiam pilot gate

RotNet now has milestone exports and a five-epoch pilot gate with the intended
1,000-epoch scheduler horizon; stopping a pilot must not change that planned
horizon. Check B=256, the anchor B=1024, then B=4096 on an appropriate GPU.
For each, record stable seconds/epoch after startup, steps/epoch, peak allocated
and reserved VRAM, finite loss/gradients, and checkpoint save/load behavior.
Test a controlled restart from a saved epoch without losing optimizer/scheduler,
scaler or RNG state. A short probe demonstrates plumbing, not converged accuracy.

For a given batch, test the three requested peak rates through warmup before
launching all corresponding long jobs. Run the SimSiam collapse check and a short
restart trial as well. A pilot on one GPU type does not establish another GPU's
memory or throughput characteristics.

### Stage D: launch the 30 Part 1 trajectories and dependent probes

RotNet's 15-task array is prepared and dependency-gated by its verified pilot CLI.
SimCLR's separate array still needs its trainer implementation. Start with at
most two concurrent GPU tasks, subject to account limits and measured resource use. Run multiple independent configurations across GPUs, rather than
allocating multiple GPUs to an unverified single-GPU script.

Each task needs a unique run folder, encoder milestone paths and last-state path.
One immutable run manifest records settings and source SHA. Save a resumable full
state at least every epoch and retain encoder snapshots every 100 epochs. Only
write a success marker after all requested epochs and artifacts are verified.

The current results logger and subset creation are not designed for simultaneous
schema migrations/first creation. Before parallel evaluation, precreate subsets
and implement per-run result files plus a serial validated merge, or tested locking.
Until then use a single evaluation writer. The existing serial downstream array
can run in a separate checkout with its own table; merge tables explicitly later.

Probe jobs can depend on the whole training array finishing, or poll only for
atomically published milestone files if that mechanism is implemented and tested.
The current RotNet submission queues pretraining only. Run frozen linear
evaluations after training; they are not automatically submitted by this launcher.

### Stage E: Part 2

Designate one SimCLR and one RotNet encoder from Part 1 without choosing on final
test scores. Train the single SimSiam encoder with the same chosen batch/epoch
setting, and fit its final frozen linear probe. Assemble the main three-method
linear table using that score and the two designated milestone scores. Then run
the nine fine-tunes and three supervised baselines in the Part 2 table. Reuse
compatible supervised rows if already completed. No further seeds or ablations
are scheduled.

## 6. Runtime and fallback budget

Do not infer hours from the RotNet log: it has no per-epoch timing and its four
rotations differ from SimCLR's two views. For 1,000 full epochs with drop_last:

| Global B | Updates/epoch | Updates per trajectory | Peak rates for base 0.5 / 1.0 / 1.5 |
|---:|---:|---:|---|
| 256 | 195 | 195,000 | 0.5 / 1.0 / 1.5 |
| 512 | 97 | 97,000 | 1 / 2 / 3 |
| 1024 | 48 | 48,000 | 2 / 4 / 6 |
| 2048 | 24 | 24,000 | 4 / 8 / 12 |
| 4096 | 12 | 12,000 | 8 / 16 / 24 |

These counts are for our specified drop_last convention; an implementation that
keeps the last partial batch has different counts and potentially a different
negative pool. Log the actual convention and draws per epoch.

Given measured seconds/epoch `t[B, recipe, GPU]`, total training GPU-hours are
`sum(1000 * t / 3600 * GPUs_per_run)` over the 30 Part 1 tasks, plus the single SimSiam pretraining run. Add probe/extraction time,
checkpoint overhead and a contingency based on pilot variability. Ideal elapsed
time is total work divided by concurrency only when runtimes/resources are
comparable; queue delays and task imbalance can dominate. Set walltimes from
measurements and verify the live partition limit before submission. If one job
exceeds it, use tested full-state continuation with the original scheduler horizon.

Fallback, in order: preserve one successful anchor and all required method/label
comparisons; keep the three smaller batches before adding 2048/4096; retain a
documented reduced set of epoch probes if necessary. For example, B={256,1024},
rates={0.5,1.0,1.5}, T=1000, probes={100,300,600,1000} gives six trajectories and
24 probes. This satisfies the proposal's reduced-grid intent; it is a contingency,
not a silent replacement for the requested 15 trajectories per method. Apply any
reduced grid consistently to SimCLR and RotNet and confirm a change of scope
before dropping planned configurations.

Do not start a 150-pretraining-run schedule-duration grid unless its larger compute
budget is an intentional team decision. It is not required by the local proposal
or the topic sheet's hardware-feasible wording.

## 7. Results, report and dates

Retain original logs and checkpoints, all individual scores, source/configuration
metadata, subset identities, errors/retries, and GPU time. A failed/OOM run is not
a zero-accuracy observation; show missing configurations explicitly. Back up final
artifacts outside transient cluster scratch storage.

Produce these deliverables:

1. Part 1: Figure B.7-style plots for SimCLR and RotNet, with learning-rate SD
   and captions naming ResNet-20, snapshot protocol and adapted probe.
2. Main frozen-linear accuracy table: SimCLR, RotNet and SimSiam at the selected
   common batch/epoch setting, using the same probe and one encoder per method.
3. Part 2: accuracy versus label fraction, comparing the three fine-tuned methods
   and supervised baseline. These are single-seed results in the current scope.
4. Supporting configuration and compute/memory table. No extra ablation figures
   are required by the current two-part plan.

The report compares batch-size/epoch behavior only for SimCLR and RotNet. It
does not conclude that SimSiam is less dependent on large batches; that question
requires measurements at multiple SimSiam batch sizes and is deferred.

| Date | Concrete milestone |
|---|---|
| Nov 16 | Proposal's check-in target: completed RotNet artifact, first downstream jobs, Git/code status, pilot measurements and explicit remaining work |
| Nov 18 | First useful SimCLR and SimSiam frozen-feature results; sweep progressing |
| Nov 25 | Complete the Part 2 1%/10%/100% comparison if measured compute permits |
| Dec 2 | Freeze core experiments and analysis, draft figures/report/poster |
| Dec 9, 1–4 pm | In-person presentation per local course guidelines |

These are planning targets, not claims of completed work. Verify check-in/report
submission times and any revised course dates on Canvas; the local guidelines
specify the presentation time but do not establish a final-report upload deadline.

## 8. Remaining work before declaring this plan fully reconciled

- Implement the missing SimCLR and SimSiam trainers against the verified shared
  baseline. The user confirmed no new push after the successful fetch. If new
  teammate work arrives later, review it before integrating or duplicating it.
- PDF retrieval is complete. Keep the verified catalog in `papers/`; use
  `python3 scripts/download_papers.py --check-only` to recheck local availability.
- Confirm the team's available GPU/account and measured pilot times.
- RotNet GPU pilot/array pipeline: done (2026-11-18). Remaining for RotNet: let
  linear-probe job `62101838` finish, merge, and build the Figure B.7 table/plot.
  SimCLR scaling still needs trainer validation.
- Add the missing trainer/orchestration capabilities only where the fetched code
  does not already provide them. Keep those implementation changes separate from
  this research/planning pass.

## Queue-priority policy: modest preference, normal Slurm scheduling

Keep RotNet's normal Nice=0 and set currently pending owned ECE662
`cifar10_lrms3...` training jobs to Nice=100. This is a mild scheduling bias,
not a strict guarantee that every RotNet task runs before every ECE662 task.
Let Slurm consider resources, job age, fair share and partition rules. Do not
hold ECE662 until all 15 models finish. Do not retain the excessive Nice=1000000
adjustment. The user reports running the previous restore command; the new
Nice=100 target is not yet confirmed remotely.

Only pending training jobs are targeted. Running tasks, the `lrm_seeds`
supervisor, RotNet's array limit and its current allocations remain unchanged.
This one-time adjustment does not apply automatically to future supervisor
submissions. No scheduler changes were made by this local plan update.
