#!/bin/bash
# Run on the Mac. Upload the prepared bundle and submit pretraining only.
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
bundle="$project_root/artifacts/rotnet_sweep_dcc.tar.gz"
test -s "$bundle"
scp "$bundle" yl1064@dcc-login.oit.duke.edu:ece661-rotnet-sweep.tar.gz
ssh yl1064@dcc-login.oit.duke.edu 'bash -s' <<'REMOTE'
set -euo pipefail
run_root="/work/yl1064/ece661-rotnet-sweep"
mkdir -p "$run_root"
tar -xzf "$HOME/ece661-rotnet-sweep.tar.gz" -C "$run_root"
cd "$run_root"
# Reuse the original CIFAR data when available; otherwise preflight downloads it once.
if [ ! -e data ] && [ -d /work/yl1064/ece661-project4/data ]; then
  ln -s /work/yl1064/ece661-project4/data data
fi
if [ -z "${ECE661_PYTHON:-}" ]; then
  for conda_root in "$HOME/miniforge3" "$HOME/miniconda" "$HOME/miniconda3" "$HOME/anaconda3"; do
    candidate="$conda_root/envs/ece661-project4/bin/python"
    if [ -x "$candidate" ]; then
      export ECE661_PYTHON="$candidate"
      break
    fi
  done
fi
if [ -z "${ECE661_PYTHON:-}" ]; then
  echo "Could not find the ece661-project4 environment. Set ECE661_PYTHON to its Python path on DCC." >&2
  exit 1
fi
echo "DCC project: $run_root"
echo "Training Python: $ECE661_PYTHON"
"$ECE661_PYTHON" scripts/submit_rotnet_sweep.py
squeue -u yl1064 -o '%.24i %.18P %.30j %.2t %.12l %.35R'
REMOTE
