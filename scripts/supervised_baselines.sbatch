#!/bin/bash
# Part 2 supervised baselines: 1%, 10%, 100% labels, seed/subset seed 0.
# Copy to the DCC project's scripts/ directory, then submit from that project:
#   mkdir -p slurm-logs
#   sbatch scripts/supervised_baselines.sbatch
# No pretrained checkpoint is required. %1 serializes shared CSV/subset writes.
# Do not overlap with other downstream writers in the same checkout.
# Four hours per task is an initial request, not a measured runtime.
#SBATCH --job-name=p4-supervised
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:2080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%1
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --exclude=dcc-gehmlab-gpu-ferc-s-z25-18,dcc-carlsonlab-gpu-ferc-s-h36-24,dcc-core-gpu-ferc-s-h36-6,dcc-majoroslab-gpu-ferc-a-r32-34

set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ece661-project4
export PYTHONNOUSERSITE=1
cd "/work/$USER/ece661-project4"

echo "Node: $(hostname) | array task: ${SLURM_ARRAY_TASK_ID}"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0))'

case "$SLURM_ARRAY_TASK_ID" in
  0) fraction=0.01 ;;
  1) fraction=0.1 ;;
  2) fraction=1.0 ;;
  *) echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

python -u train_classifier.py \
  --fraction "$fraction" \
  --epochs 100 \
  --batch-size 128 \
  --lr 0.1 \
  --weight-decay 5e-4 \
  --seed 0 \
  --subset-seed 0 \
  --num-workers 4 \
  --min-epoch-size 5000 \
  --eval-every 10 \
  --save "checkpoints/supervised_fraction${fraction}_seed0_${SLURM_ARRAY_JOB_ID}.pt"
