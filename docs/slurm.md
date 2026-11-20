# Running jobs on DCC

The [RotNet sweep and its frozen probes are complete](../results/rotnet/README.md).
Use the Part 2 job for the limited-label comparison. Use the sweep commands
below for a new campaign.

## Run Part 2

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
