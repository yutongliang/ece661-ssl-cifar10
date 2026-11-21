#!/bin/bash
# RotNet ResNet-50 pretraining: task 0 = batch 256 (LR 0.1), task 1 = batch 1024
# (LR 0.4). 1,000 epochs, cosine, encoder export every 100 epochs. Set the
# gradient-accumulation splits measured by r50_pilot.sbatch:
#   sbatch --export=ALL,ACCUM_256=1,ACCUM_1024=4 scripts/slurm/r50_rotnet_pretrain.sbatch
# The run saves its full state every epoch. Ten minutes before the time limit the
# task requeues itself and resumes from the last completed epoch.
#SBATCH --job-name=p4-r50-rotnet
#SBATCH --partition=gpu-common,scavenger-gpu
#SBATCH --gres=gpu:5000_ada:1
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
#SBATCH --exclude=dcc-gehmlab-gpu-ferc-s-z25-18,dcc-carlsonlab-gpu-ferc-s-h36-24,dcc-core-gpu-ferc-s-h36-6,dcc-majoroslab-gpu-ferc-a-r32-34

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python_path="${ECE661_PYTHON:-$HOME/miniforge3/envs/ece661-project4/bin/python}"
export PYTHONNOUSERSITE=1
: "${ACCUM_256:?set ACCUM_256 from the pilot}" "${ACCUM_1024:?set ACCUM_1024 from the pilot}"

batches=(256 1024); lrs=(0.1 0.4); accums=("$ACCUM_256" "$ACCUM_1024"); tags=(0p1 0p4)
task=$SLURM_ARRAY_TASK_ID
out="checkpoints/r50/rotnet_r50_b${batches[$task]}_lr${tags[$task]}_seed0/encoder.pt"
resume=()
[ -f "$out.last" ] && resume=(--resume)
echo "Node: $(hostname) | task $task | restart ${SLURM_RESTART_COUNT:-0} | resume ${resume[*]:-no}"

requeue() { echo "Time limit near: requeueing $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; }
trap requeue USR1
"$python_path" -u -m cifar_ssl.pretrain_rotnet --arch resnet50 --epochs 1000 \
  --batch-size "${batches[$task]}" --lr "${lrs[$task]}" --accum-steps "${accums[$task]}" \
  --weight-decay 5e-4 --seed 0 --num-workers 8 --eval-every 10 --checkpoint-every 100 \
  --require-cuda --out "$out" "${resume[@]}" &
wait $!
