#!/bin/bash
# ResNet-50 GPU pilot: measure one RotNet epoch at batch 256 and 1024 (with the
# smallest gradient-accumulation split that fits), then time one fine-tune and
# one linear-eval epoch. Writes only under pilot/. Run from the ResNet-50 root:
#   sbatch scripts/slurm/r50_pilot.sbatch
#SBATCH --job-name=p4-r50-pilot
#SBATCH --partition=gpu-common,scavenger-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --exclude=dcc-gehmlab-gpu-ferc-s-z25-18,dcc-carlsonlab-gpu-ferc-s-h36-24,dcc-core-gpu-ferc-s-h36-6,dcc-majoroslab-gpu-ferc-a-r32-34

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
python_path="${ECE661_PYTHON:-$HOME/miniforge3/envs/ece661-project4/bin/python}"
export PYTHONNOUSERSITE=1
echo "Node: $(hostname)"
"$python_path" -c 'import torch; assert torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name(0))' || exit 1
rm -rf pilot && mkdir -p pilot

declare -A chosen
for batch in 256 1024; do
  lr=$("$python_path" -c "print(0.1 * $batch / 256)")
  for accum in 1 2 4 8 16; do
    (( batch % accum == 0 )) || continue
    out="pilot/rotnet_b${batch}_accum${accum}/encoder.pt"
    echo "=== RotNet ResNet-50 | batch $batch | lr $lr | accum $accum"
    if "$python_path" -u -m cifar_ssl.pretrain_rotnet --arch resnet50 --epochs 1000 \
         --stop-after-epoch 1 --checkpoint-every 1 --batch-size "$batch" --lr "$lr" \
         --accum-steps "$accum" --eval-every 1000 --num-workers 8 --require-cuda --out "$out"; then
      chosen[$batch]=$accum
      echo "RESULT batch=$batch accum=$accum fits"
      break
    fi
    echo "RESULT batch=$batch accum=$accum failed (likely out of memory); trying a larger split"
  done
done

encoder=pilot/rotnet_b256_accum${chosen[256]:-1}/encoder_epoch0001.pt
if [ -f "$encoder" ]; then
  echo "=== Fine-tune epoch | 1% | batch 256"
  "$python_path" -u -m cifar_ssl.downstream --init "$encoder" --fraction 0.01 --batch-size 256 \
    --lr 0.05 --max-epochs 2 --num-workers 8 --out-dir pilot/finetune
  echo "=== Linear-eval epoch | 100% | batch 1024"
  "$python_path" -u -m cifar_ssl.downstream --init "$encoder" --freeze-encoder --fraction 1.0 \
    --batch-size 1024 --lr 0.4 --max-epochs 1 --num-workers 8 --out-dir pilot/linear
fi
echo "Chosen splits: batch256=${chosen[256]:-none} batch1024=${chosen[1024]:-none}"
