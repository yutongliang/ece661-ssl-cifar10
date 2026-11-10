#!/bin/bash
# RotNet 2-epoch GPU pilot on DCC.
# Usage, from /work/$USER/ece661-project4:
#   mkdir -p slurm-logs        # Slurm does not create the log folder
#   sbatch scripts/rotnet_pilot.sbatch
# Assumes Miniforge at $HOME/miniforge3 and a conda env named ece661-project4;
# edit the "source"/"conda activate" lines if yours differ.
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
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ece661-project4
export PYTHONNOUSERSITE=1
cd "/work/$USER/ece661-project4"

echo "Node: $(hostname)"
python -m unittest discover -s tests -v
python -c "import torch; assert torch.cuda.is_available(), \"CUDA unavailable\"; print(\"GPU:\", torch.cuda.get_device_name(0))"

python -u pretrain_rotnet.py \
  --epochs 2 \
  --batch-size 128 \
  --num-workers 4 \
  --eval-every 1 \
  --seed 0 \
  --out "checkpoints/pilot_rotnet_seed0_${SLURM_JOB_ID}.pt"
