# Running jobs on DCC

The ResNet-50 comparison ([plan](experiments.md)) runs from its own DCC folder,
`/work/yl1064/ece661-r50`, which holds this checkout, the shared
[subsets](../subsets/), and a `data` link to the existing CIFAR-10 download. The
[ResNet-20 RotNet study](../results/rotnet/README.md) is complete; its commands
follow in the later sections.

## ResNet-50 RotNet

Run every command from the ResNet-50 folder. The scripts use `ECE661_PYTHON`
when set, otherwise the Miniforge `ece661-project4` environment.

| Step | Command | What it does |
|---|---|---|
| 1. Pilot | `sbatch scripts/slurm/r50_pilot.sbatch` | One RotNet epoch at batch 256 and 1024, trying gradient-accumulation splits 1, 2, 4, 8, 16 until one fits; then one fine-tune and one linear-eval epoch. Writes only `pilot/` |
| 2. Pretrain | `sbatch --export=ALL,ACCUM_256=N,ACCUM_1024=M scripts/slurm/r50_rotnet_pretrain.sbatch` | Tasks 0 and 1: batch 256 (LR 0.1) and 1024 (LR 0.4), 1,000 epochs, encoders every 100 epochs in `checkpoints/r50/<run>/`. Use the splits the pilot reports |
| 3. Linear eval | `sbatch --export=ALL,MODE=linear --array=0-20%4 scripts/slurm/r50_downstream.sbatch` | 20 checkpoints plus the random-init control; frozen encoder, 100% labels |
| 4. Fine-tune | `sbatch --export=ALL,MODE=finetune --array=0-7%4 scripts/slurm/r50_downstream.sbatch` | Both epoch-1,000 encoders at 1, 3, 5, and 10% labels |
| 5. Collect | `python scripts/collect_r50.py` | Merges `results/r50/*/**/result.json` into `results/r50/linear.csv` and `finetune.csv`; lists missing runs |

- **Resuming:** pretraining and downstream runs save their full state every epoch.
  Ten (pretraining) or five (downstream) minutes before the time limit, a task
  requeues itself and resumes from its last completed epoch. Scavenger
  preemption resumes the same way.
- **Skipping:** a downstream task whose `result.json` exists with the same settings
  is skipped, so resubmitting an array only runs what is missing.
- **Batch norm with accumulation:** gradient accumulation matches the full-batch
  gradient, but batch norm normalizes each sub-batch.
- **Order:** submit linear eval and fine-tuning after pretraining has written the
  encoders they need, or add `--dependency=afterok:<pretrain job>`.

## Run Part 2 (ResNet-20)

On DCC, use a checkout containing the current Python package, Slurm scripts,
and [saved label subsets](../subsets/). Run from its root. The batch scripts
use `ECE661_PYTHON` when set, otherwise they use the existing Miniforge
`ece661-project4` environment.

```bash
mkdir -p slurm-logs
sbatch scripts/slurm/rotnet_part2.sbatch
```

Part 2 checks the published anchor at
`checkpoints/figure_b7/rotnet_r20_b1024_lr0p1_seed0/encoder_epoch1000.pt` and both
subset checksums before training. Each group uses 1%, 10%, then 100% of the labels.
Job `62251891` ran this default. The extra fractions (5%, 8% = the common
4,000-label setting, 20%) use the same script after the default job finishes,
because both write the same CSV:

```bash
sbatch --array=0-8%1 --export=ALL,FRACTIONS="0.05 0.08 0.2" \
  --dependency=afterany:62251891 scripts/slurm/rotnet_part2.sbatch
```

With n fractions, tasks 0..n-1 fine-tune, n..2n-1 train from scratch, and
2n..3n-1 fit frozen probes. The table below is the default n = 3.

| Array tasks | Fit | Output |
|---|---|---|
| 0–2 | Fine-tune RotNet from the anchor | `results/results.csv` and `checkpoints/part2/` |
| 3–5 | Train from scratch | `results/results.csv` and `checkpoints/part2/` |
| 6–8 | Fit a frozen linear classifier | `results/results.csv` |

The array runs one task at a time because the CSV has a single writer.
Run standalone downstream jobs separately in the same checkout. On requeue,
a Part 2 task starts its fit from epoch 1. Use the Slurm task ID and run settings
to check for duplicate CSV rows if you retry manually.

## Launch a new RotNet sweep

From a fresh DCC run directory with the current checkout and training environment:

```bash
python scripts/slurm/rotnet_sweep.py validate
python scripts/slurm/submit_rotnet_sweep.py --dry-run
python scripts/slurm/submit_rotnet_sweep.py
```

| Stage | What it does |
|---|---|
| CPU preflight | Checks the pinned environment, runs synthetic tests, and downloads CIFAR-10 once |
| GPU pilots | Runs five epochs for all 15 configurations with the original 1,000-epoch schedule. Checks finite loss and the timing budget. |
| GPU array | Continues the same runs to epoch 1000 after the pilots pass. Defaults to `gpu-common`, `5000_ada`, and two concurrent tasks. |
| Frozen probes | Submit `sbatch scripts/slurm/rotnet_linear_probes.sbatch` after training. Tasks 0–14 each fit ten probes. Task 15 uses random features. |

Merge the 151 per-probe JSON outputs with
`python scripts/slurm/rotnet_linear_probes.py merge`. The merge reports missing
probes and writes `results/rotnet_linear_probes.csv`. Completed probes with the
same checkpoint hash and protocol are reused on retry.

A submission writes its receipt and checksummed source under
`results/rotnet_sweep_submissions/`. The dry run writes a local receipt with
simulated IDs. The live submission checks queue capacity and active campaigns
before sending jobs. Monitor through `squeue -u "$USER"`, then inspect
`slurm-logs/` or use `sacct -j JOB_ID` when a job exits. If a dependency fails,
start with that stage's log.

Each sweep run saves full training state to `encoder.pt.last` after each epoch.
Use the matching frozen submission source to resume an older sweep. A new source
hash requires a fresh run directory. For standalone RotNet, keep the original
total `--epochs` and training settings when using `--resume`. The `.last` file
restores the model, optimizer, scheduler, scaler, and random state. Use the
`.pt` encoder export for evaluation and fine-tuning.

## Other cluster tools

| Command | Use |
|---|---|
| `sbatch scripts/slurm/rotnet_pilot.sbatch` | Two-epoch GPU smoke check |
| `python scripts/slurm/dcc_job_status.py` | Inspect the account's queue on DCC |
| `bash scripts/slurm/launch_rotnet_dcc.sh` | Team-account shortcut from the Mac. Packages this checkout, uploads to the configured sweep root, and submits the pretraining stages. |
