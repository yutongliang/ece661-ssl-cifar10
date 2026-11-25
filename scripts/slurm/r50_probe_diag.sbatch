#!/bin/bash
# Diagnose the late drop in RotNet linear-eval accuracy (docs/RotNet.md, to-do 4).
#   task 0: L-BFGS logistic regression on standardized frozen features (layer4 and
#           layer3) for all 20 checkpoints -> results/r50/diagnostics/lbfgs.csv
#   task 1: SGD linear eval, batch 256, epoch-300 encoder, cap 300 epochs instead of 100
#   task 2: same for the epoch-1000 encoder
# Tasks 1-2 use the shared recipe otherwise unchanged (LR 0.1, plateau rule).
#   bash scripts/slurm/r50_sbatch.sh --array=0-2 scripts/slurm/r50_probe_diag.sbatch
#SBATCH --job-name=p4-r50-diag
#SBATCH --partition=gpu-common,scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yl1064@duke.edu
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --open-mode=append

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python_path="${ECE661_PYTHON:-$HOME/miniforge3/envs/ece661-project4/bin/python}"
export PYTHONNOUSERSITE=1
task=$SLURM_ARRAY_TASK_ID
echo "Node: $(hostname) | diag task $task | restart ${SLURM_RESTART_COUNT:-0}"
source scripts/slurm/r50_gpu_check.sh

requeue() { echo "Time limit near: requeueing $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; }
trap requeue USR1
case "$task" in
  0)
    "$python_path" -u scripts/probe_diagnostic.py --out results/r50/diagnostics/lbfgs.csv \
      checkpoints/r50/rotnet_r50_b256_lr0p1_seed0/encoder_epoch{0100,0200,0300,0400,0500,0600,0700,0800,0900,1000}.pt \
      checkpoints/r50/rotnet_r50_b1024_lr0p4_seed0/encoder_epoch{0100,0200,0300,0400,0500,0600,0700,0800,0900,1000}.pt &
    ;;
  1|2)
    epoch=$([ "$task" = 1 ] && echo 0300 || echo 1000)
    run=rotnet_r50_b256_lr0p1_seed0
    "$python_path" -u -m cifar_ssl.downstream --init "checkpoints/r50/$run/encoder_epoch$epoch.pt" \
      --freeze-encoder --fraction 1.0 --batch-size 256 --lr 0.1 \
      --seed 0 --subset-seed 0 --num-workers 8 --max-epochs 300 --plateau-patience 10 \
      --plateau-threshold 0.01 --lr-drops 1 --weight-decay 0 --eval-every 10 \
      --out-dir "results/r50/diagnostics/sgd300/$run/epoch$epoch" &
    ;;
  *) echo "Unexpected task $task" >&2; exit 1 ;;
esac
wait $!
