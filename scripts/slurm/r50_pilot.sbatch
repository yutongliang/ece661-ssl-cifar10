#!/bin/bash
# ResNet-50 GPU pilot, capped at 22 GiB so its choices fit every allowed GPU.
# 1. Largest RotNet sub-batch (original images, x4 after rotation) that fits.
# 2. Stability: 12 epochs (past the 10-epoch warmup) at batch 256 and 1024, plus
#    5 epochs at batch 256 without warmup for comparison.
# 3. Largest fine-tune sub-batch (at most 256) at batch 1024; one linear-eval epoch.
# 4. Only if every gate passes, submit the two 1,000-epoch pretraining runs.
# Writes only under pilot/ (and the pretraining job, if released). Submit with:
#   bash scripts/slurm/r50_sbatch.sh scripts/slurm/r50_pilot.sbatch
#SBATCH --job-name=p4-r50-pilot
#SBATCH --partition=gpu-common,scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yl1064@duke.edu
#SBATCH --output=slurm-logs/%x-%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
python_path="${ECE661_PYTHON:-$HOME/miniforge3/envs/ece661-project4/bin/python}"
export PYTHONNOUSERSITE=1
echo "Node: $(hostname)"
source scripts/slurm/r50_gpu_check.sh
cap=(--max-gpu-memory-gb 22)
rm -rf pilot && mkdir -p pilot

# 0 = success, 2 = out of memory, 3 = non-finite loss; anything else stops the pilot.
attempt() {
  local log; log=$(mktemp)
  "$@" 2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  local code=1
  if (( status == 0 )); then code=0
  elif grep -qE "OutOfMemoryError|CUDA out of memory" "$log"; then code=2
  elif grep -q "Non-finite" "$log"; then code=3
  fi
  rm -f "$log"
  (( code == 1 )) && { echo "PILOT ERROR: failed for a reason other than memory or divergence"; exit 1; }
  return $code
}

rotnet() {  # batch micro warmup stop_epoch tag
  "$python_path" -u -m cifar_ssl.pretrain_rotnet --arch resnet50 --epochs 1000 --warmup-epochs "$3" \
    --stop-after-epoch "$4" --checkpoint-every "$4" --batch-size "$1" \
    --lr "$("$python_path" -c "print(0.1 * $1 / 256)")" --accum-steps $(( $1 / $2 )) \
    --eval-every 1000 --num-workers 8 --require-cuda "${cap[@]}" --out "pilot/$5/encoder.pt"
}

micro=""
for m in 256 128 64 32; do
  echo "=== RotNet sub-batch $m images ($(( 4 * m )) rotated) at batch 256"
  attempt rotnet 256 "$m" 10 1 "search_micro$m"; rc=$?
  if (( rc == 0 )); then micro=$m; echo "RESULT rotnet_micro=$m fits"; break; fi
  echo "RESULT rotnet_micro=$m does not fit (code $rc)"
done
[ -n "$micro" ] || { echo "GATE FAIL: no RotNet sub-batch fits in 22 GiB"; exit 1; }

for spec in "256 10 12 b256_warmup" "1024 10 12 b1024_warmup" "256 0 5 b256_nowarmup"; do
  set -- $spec
  echo "=== Stability | batch $1 | warmup $2 | $3 epochs"
  attempt rotnet "$1" "$micro" "$2" "$3" "$4"; echo "RESULT $4 exit code $?"
done

# Gate: both warmup runs finished 12 finite epochs and learned the rotation task.
"$python_path" - <<'PY'
import math, sys, torch
ok = True
for tag in ("b256_warmup", "b1024_warmup", "b256_nowarmup"):
    try:
        history = torch.load(f"pilot/{tag}/encoder.pt.last", weights_only=True)["history"]
    except FileNotFoundError:
        history = []
    last = history[-1] if history else None
    line = (f"{tag}: {len(history)} epochs" + (f", final loss {last['loss']:.3f}, rotation acc "
            f"{last['rotation_acc']:.3f}, {last['seconds']:.0f} s/epoch" if last else ""))
    print("STABILITY", line)
    if tag.endswith("_warmup"):
        good = (len(history) == 12 and all(math.isfinite(h["loss"]) for h in history)
                and last["loss"] < 1.0 and last["rotation_acc"] > 0.5)
        ok &= good
        print("GATE", "PASS" if good else "FAIL", tag)
sys.exit(0 if ok else 1)
PY
gate=$?

encoder="pilot/b256_warmup/encoder_epoch0012.pt"
ft_micro=""
if [ -f "$encoder" ]; then
  for m in 256 128 64; do
    echo "=== Fine-tune sub-batch $m at batch 1024 (1% labels)"
    attempt "$python_path" -u -m cifar_ssl.downstream --init "$encoder" --fraction 0.01 --batch-size 1024 \
      --accum-steps $(( 1024 / m )) --lr 0.2 --max-epochs 2 --num-workers 8 "${cap[@]}" \
      --out-dir "pilot/finetune_micro$m"; rc=$?
    if (( rc == 0 )); then ft_micro=$m; echo "RESULT finetune_micro=$m fits"; break; fi
    echo "RESULT finetune_micro=$m does not fit (code $rc)"
  done
  echo "=== Linear-eval epoch | 100% | batch 1024"
  attempt "$python_path" -u -m cifar_ssl.downstream --init "$encoder" --freeze-encoder --fraction 1.0 \
    --batch-size 1024 --lr 0.4 --max-epochs 1 --num-workers 8 "${cap[@]}" --out-dir pilot/linear
fi
echo "Chosen: ROTNET_MICRO=$micro FT_MICRO=${ft_micro:-none}"

if (( gate == 0 )) && [ -n "$ft_micro" ]; then
  echo "=== All gates passed: submitting the 1,000-epoch pretraining runs"
  bash scripts/slurm/r50_sbatch.sh --export=ALL,MICRO="$micro" scripts/slurm/r50_rotnet_pretrain.sbatch
else
  echo "GATE FAIL: pretraining not submitted"
  exit 1
fi
