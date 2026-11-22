#!/bin/bash
# RotNet ResNet-50 pretraining: task 0 = batch 256 (LR 0.1), task 1 = batch 1024
# (LR 0.4). 1,000 epochs, 10-epoch linear warmup then cosine (SimCLR 2.3), encoder
# export every 100 epochs. MICRO is the
# sub-batch in original images chosen by r50_pilot.sbatch; both runs use it, so
# batch norm sees the same sub-batch whichever GPU a task lands on:
#   bash scripts/slurm/r50_sbatch.sh --export=ALL,MICRO=64 scripts/slurm/r50_rotnet_pretrain.sbatch
# The run saves its full state every epoch. Ten minutes before the time limit the
# task requeues itself and resumes from the last completed epoch.
#SBATCH --job-name=p4-r50-rotnet
#SBATCH --partition=gpu-common,scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yl1064@duke.edu
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --open-mode=append

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python_path="${ECE661_PYTHON:-$HOME/miniforge3/envs/ece661-project4/bin/python}"
export PYTHONNOUSERSITE=1
: "${MICRO:?set MICRO (sub-batch in original images) from the pilot}"

batches=(256 1024); lrs=(0.1 0.4); tags=(0p1 0p4)
task=$SLURM_ARRAY_TASK_ID
out="checkpoints/r50/rotnet_r50_b${batches[$task]}_lr${tags[$task]}_seed0/encoder.pt"
resume=()
[ -f "$out.last" ] && resume=(--resume)
echo "Node: $(hostname) | task $task | restart ${SLURM_RESTART_COUNT:-0} | resume ${resume[*]:-no}"
source scripts/slurm/r50_gpu_check.sh
accum=$(( ${batches[$task]} / MICRO ))

requeue() { echo "Time limit near: requeueing $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; }
trap requeue USR1
"$python_path" -u -m cifar_ssl.pretrain_rotnet --arch resnet50 --epochs 1000 \
  --batch-size "${batches[$task]}" --lr "${lrs[$task]}" --accum-steps "$accum" \
  --weight-decay 5e-4 --warmup-epochs 10 --seed 0 --num-workers 8 --eval-every 10 --checkpoint-every 100 \
  --require-cuda --out "$out" "${resume[@]}" &
wait $!
