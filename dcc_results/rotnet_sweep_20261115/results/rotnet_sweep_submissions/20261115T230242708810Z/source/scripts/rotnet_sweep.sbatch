#!/bin/bash
# Resources/dependencies are supplied by submit_rotnet_sweep.py.
# Requeued tasks resume the same configuration and total schedule.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
cd "${ECE661_SOURCE:-$SLURM_SUBMIT_DIR}"
export PYTHONNOUSERSITE=1
python_path="${ECE661_PYTHON:-$HOME/miniforge3/envs/ece661-project4/bin/python}"
test -x "$python_path"
echo "Job: $SLURM_JOB_ID | phase: $1 | node: $(hostname) | restart: ${SLURM_RESTART_COUNT:-0}"
case "$1" in
  preflight) "$python_path" -u scripts/rotnet_sweep.py preflight ;;
  pilots) "$python_path" -u scripts/rotnet_sweep.py pilots ;;
  train) "$python_path" -u scripts/rotnet_sweep.py run --index "$SLURM_ARRAY_TASK_ID" ;;
  *) echo "Unknown phase: $1" >&2; exit 1 ;;
esac
