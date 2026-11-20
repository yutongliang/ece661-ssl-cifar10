#!/bin/bash
# Part 1 probes. Tasks 0–14 cover ten snapshots each; task 15 is the random control.
#SBATCH --job-name=p4-rotnet-probe
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-15%4
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --exclude=dcc-gehmlab-gpu-ferc-s-z25-18,dcc-carlsonlab-gpu-ferc-s-h36-24,dcc-core-gpu-ferc-s-h36-6,dcc-majoroslab-gpu-ferc-a-r32-34

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONNOUSERSITE=1
python_path="${ECE661_PYTHON:-$HOME/miniforge3/envs/ece661-project4/bin/python}"
test -x "$python_path"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | node: $(hostname) | restart: ${SLURM_RESTART_COUNT:-0}"
"$python_path" -u scripts/slurm/rotnet_linear_probes.py run --task "$SLURM_ARRAY_TASK_ID"
