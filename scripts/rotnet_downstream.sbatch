#!/bin/bash
# First downstream comparison: 2 linear probes + 6 classifier runs, seed 0.
# Run from /work/$USER/ece661-project4 after copying this script there:
#   mkdir -p slurm-logs
#   sbatch scripts/rotnet_downstream.sbatch
# %1 serializes this array's tasks because results.csv and subsets are shared.
# Avoid overlapping this array with other downstream writers in this checkout.
# The 4-hour limit is an initial per-task request, not a measured runtime.
#SBATCH --job-name=p4-rotnet-eval
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:2080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-7%1
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --exclude=dcc-gehmlab-gpu-ferc-s-z25-18,dcc-carlsonlab-gpu-ferc-s-h36-24,dcc-core-gpu-ferc-s-h36-6,dcc-majoroslab-gpu-ferc-a-r32-34

set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ece661-project4
export PYTHONNOUSERSITE=1
cd "/work/$USER/ece661-project4"

echo "Node: $(hostname) | array task: ${SLURM_ARRAY_TASK_ID}"
test -s checkpoints/rotnet.pt
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0))'

common=(--epochs 100 --seed 0 --subset-seed 0 --num-workers 4)
case "$SLURM_ARRAY_TASK_ID" in
  0)
    python -u linear_eval.py --checkpoint checkpoints/rotnet.pt \
      --fraction 1.0 --stage 3 --pool 1 "${common[@]}"
    ;;
  1)
    python -u linear_eval.py --random-init \
      --fraction 1.0 --stage 3 --pool 1 "${common[@]}"
    ;;
  [2-7])
    fractions=(0.01 0.1 1.0)
    fraction="${fractions[$(( (SLURM_ARRAY_TASK_ID - 2) % 3 ))]}"
    init=()
    method=supervised
    if (( SLURM_ARRAY_TASK_ID >= 5 )); then
      method=rotnet_finetune
      init=(--init checkpoints/rotnet.pt)
    fi
    python -u train_classifier.py --fraction "$fraction" "${init[@]}" \
      "${common[@]}" --batch-size 128 --lr 0.1 --weight-decay 5e-4 \
      --min-epoch-size 5000 --eval-every 10 \
      --save "checkpoints/${method}_fraction${fraction}_seed0_${SLURM_ARRAY_JOB_ID}.pt"
    ;;
  *)
    echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2
    exit 1
    ;;
esac
