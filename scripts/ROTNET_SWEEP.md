# RotNet first: 15 pretrained models

**Status 2026-11-18: pretraining complete.** `61863269` preflight -> `61863270`
pilots -> `61863271` 15-model array (launch ID `20261115T230242708810Z`). All 15
training tasks finished COMPLETED with exit 0:0 (2.6–4.0 h each), each with a
verified `complete.json`. Outputs are copied locally to
`dcc_results/rotnet_sweep_20261115/` (encoders are git-ignored; small run records
are in `run_records/`). Do not rerun the launcher for this grid.

The accuracies logged during pretraining are 4-way **rotation** (pretext) top-1,
extracted to `dcc_results/rotnet_sweep_20261115/experiments/rotnet_pretrain_rotation_top1.csv`.
They are not CIFAR-10 class accuracies; Figure B.7 needs the frozen linear probes below.

## Frozen linear probes (CIFAR-10 class top-1) — job `62101838`

`scripts/rotnet_linear_probes.py` fits a fresh linear classifier on each of the
150 frozen milestone encoders, with exactly the `linear_eval.py` protocol (stage 3,
pool 1, 64-d, 100% labels, Adam lr 1e-3, wd 1e-5, 100 epochs, seed 0). It was
checked to reproduce `linear_eval.py`'s accuracy on the same checkpoint. Each
probe writes its own JSON (no shared `results/results.csv` writes), and completed
probes with matching checkpoint SHA256 and protocol are skipped on requeue.

Array tasks 0–14 are the 15 runs in manifest order (10 milestones each); task 15
is a random-initialization floor. Submitted 2026-11-18 from the DCC run root:

```bash
cd /work/yl1064/ece661-rotnet-sweep
sbatch scripts/rotnet_linear_probes.sbatch                      # already done: 62101838
squeue -u yl1064 -n p4-rotnet-probe --array
~/miniforge3/envs/ece661-project4/bin/python scripts/rotnet_linear_probes.py merge
```

`merge` writes `results/rotnet_linear_probes.csv` and exits non-zero listing any
missing probes. The designated main-table RotNet encoder is the prespecified
anchor `rotnet_r20_b1024_lr0p1_seed0` at epoch 1000, not the best test score.

## Original launch instructions

From the Mac, after preparing the bundle:

```bash
bash /Users/teresaliang/Desktop/ECE661/project4/scripts/launch_rotnet_dcc.sh
```

This uploads the bundle to `yl1064@dcc-login.oit.duke.edu`, extracts it into
`/work/yl1064/ece661-rotnet-sweep`, finds the existing `ece661-project4` Python
environment, and submits the pretraining campaign. It uses a separate directory
from the existing ECE661 and ECE662 jobs. No fine-tuning or baseline jobs are
submitted. Frozen linear evaluation comes after training, using the saved snapshots.

The pipeline is:

1. CPU preflight: check the pinned environment, run the synthetic tests, and prepare
   CIFAR-10 once to avoid concurrent downloads.
2. One GPU pilot job: check all 15 configurations for five epochs each, starting
   with the largest batch and rate. The original scheduler horizon stays at 1000.
3. After all pilots succeed, a 15-task GPU array continues the same trajectories
   from their saved states to epoch 1000, with at most two tasks running at once.

| Batch sizes | SGD learning rates | Total epochs | Encoder snapshots |
|---|---|---:|---|
| 256, 512, 1024, 2048, 4096 original images | 0.03, 0.1, 0.3, no batch scaling | 1000 | Every 100 epochs |

The rates are prespecified project settings centered on the prior successful 0.1
recipe; they are not claimed optimal and the GPU pilots still need to run. Each
batch becomes four times as many images after rotation. There is no microbatch
substitution that silently changes the requested batch/BN behavior.

Default GPU request for future submissions: `gpu-common`, one `5000_ada` (32 GB)
per job. The user's live inventory shows three gpu-common nodes with eight such
GPUs each; this is installed capacity, not a count of currently free GPUs.
The active submission was originally sent to scavenger-gpu. The user has now
successfully moved both pending GPU stages to gpu-common using these commands
(already completed; no need to repeat):

```bash
scontrol update JobId=61863270 Partition=gpu-common
scontrol update JobId=61863271 Partition=gpu-common
squeue -j 61863270,61863271 -o '%.24i %.16P %.24j %.2t %.35R'
```

These commands change the queued jobs' partition, not their IDs, source or
dependencies. Do not rerun the launcher to make this change. The supplied
squeue output confirms gpu-common for both jobs; both were still PD with reason
None at that observation. This is not evidence that pilot training has started.

The pilot
checks whether the largest batch actually fits. Its two-hour request and the
48-hour full-run request are initial scheduling choices. The pilot gate refuses
to release full training if its measured estimate, with 25% margin, exceeds
48 hours. Inspect the error and revise scheduling if this happens.

All jobs exclude the four nodes recorded in ECE662's `scripts/grid_driver.py`:

```text
dcc-gehmlab-gpu-ferc-s-z25-18
dcc-carlsonlab-gpu-ferc-s-h36-24
dcc-core-gpu-ferc-s-h36-6
dcc-majoroslab-gpu-ferc-a-r32-34
```

The submitted source and checksums are frozen under
`results/rotnet_sweep_submissions/<timestamp>/source/`. Each configuration has
its own `checkpoints/figure_b7/<run_id>/` directory, metadata, lock, `encoder.pt.last`,
pilot report, ten milestone encoders, and final completion record. Later local
edits do not change already submitted source. Preemption requeues the task and
restores the last completed epoch; at most the unfinished epoch is repeated.
Training-code or parameter mismatches are rejected before reusing a run directory.

The active submission uses the original `flock` locks; one initial contention
error disappeared on retry. Later local code uses atomic directory locks with
host/PID/Slurm-owner metadata and conservative stale-owner recovery, covered by
crash/restart tests. This change is for future submissions and does not modify
the frozen source of the active jobs.

Monitor on DCC:

```bash
cd /work/yl1064/ece661-rotnet-sweep
squeue -u yl1064 -o '%.24i %.18P %.30j %.2t %.12l %.35R'
cat results/rotnet_sweep_submissions/*/submission.json
```

The submission receipt prints the preflight, pilot and training-array IDs.
If a gate fails, downstream `afterok` jobs remain blocked; inspect the gate's
Slurm log instead of treating a queued array as evidence that training started.
Re-submission refuses an existing active campaign. If a job terminates rather
than being requeued, resubmit after it leaves the queue; matching runs resume
their saved states and already completed configurations are verified and reused.

## Prioritize ahead of the existing ECE662 queue

The original launch set 290 pending ECE662 jobs to Nice=1000000. This
excessively suppressed their priority. The user reports running the restore
command and has chosen a new policy: retain RotNet's normal Nice=0, set current
pending ECE662 training jobs to Nice=100, and let Slurm schedule both projects.
There is no requirement to wait until all 15 RotNet models finish before ECE662
can run. The Nice=100 change has not yet been confirmed remotely.

Future Mac launches no longer automatically adjust other projects. An explicit
`--prioritize-ece662` flag uses a target of Nice=100 and records prior values.
The helper sets this target exactly, so a legacy Nice=1000000 is replaced by 100.
Running jobs and the supervisor are not cancelled, suspended or stopped.

This lowers the competing jobs' scheduling priority. It does not override site
limits, partition tiers or other users' priorities, and it cannot reclaim a GPU
from a currently running job. ECE662's supervisor may submit new jobs later; those
new jobs are not covered by this one-time adjustment. Review the live queue if
additional adjustments are needed.

If a later explicit decision calls for undoing a recorded adjustment, use the
matching record's path (replace the timestamp below). This is not required to
apply the currently selected Nice=100 policy:

```bash
python3 scripts/prioritize_rotnet.py --restore --apply \
  --record results/rotnet_sweep_submissions/<timestamp>/ece662_priority.json
```

Restoration only touches still-pending owned jobs whose Nice value still matches
this adjustment, preserving subsequent manual changes. No job priority has been
changed locally; Slurm changes happen only when the launch command runs on DCC.
