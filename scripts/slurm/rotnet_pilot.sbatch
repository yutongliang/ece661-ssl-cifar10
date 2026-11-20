#!/bin/bash
# Quick two-epoch smoke test on DCC. See docs/slurm.md.
#SBATCH --job-name=p4-rotnet-pilot
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:2080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-logs/%x-%j.out
# Nodes with known GPU faults (h36-6 crashed job 61349587 with cudaErrorInvalidAddressSpace)
#SBATCH --exclude=dcc-gehmlab-gpu-ferc-s-z25-18,dcc-carlsonlab-gpu-ferc-s-h36-24,dcc-core-gpu-ferc-s-h36-6,dcc-majoroslab-gpu-ferc-a-r32-34

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1
python_path="${ECE661_PYTHON:-$HOME/miniforge3/envs/ece661-project4/bin/python}"
test -x "$python_path"

echo "Node: $(hostname)"
"$python_path" -m unittest discover -s tests -v
"$python_path" -c "import torch; assert torch.cuda.is_available(), \"CUDA unavailable\"; print(\"GPU:\", torch.cuda.get_device_name(0))"

"$python_path" -u -m cifar_ssl.pretrain_rotnet \
  --epochs 2 \
  --batch-size 128 \
  --num-workers 4 \
  --eval-every 1 \
  --seed 0 \
  --out "checkpoints/pilot_rotnet_seed0_${SLURM_JOB_ID}.pt"
