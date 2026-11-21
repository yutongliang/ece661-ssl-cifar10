#!/bin/bash
# RotNet ResNet-50 downstream runs with the shared recipe (docs/experiments.md).
#   MODE=linear   tasks 0-20: 2 runs x 10 checkpoints, then the random-init control
#   MODE=finetune tasks 0-7:  2 epoch-1000 encoders x 1, 3, 5, 10% labels
#   sbatch --export=ALL,MODE=linear --array=0-20%4 scripts/slurm/r50_downstream.sbatch
#   sbatch --export=ALL,MODE=finetune --array=0-7%4 scripts/slurm/r50_downstream.sbatch
# Each task owns its output folder, so tasks may run in parallel. A requeued task
# resumes from its last epoch; a finished task is skipped.
#SBATCH --job-name=p4-r50-down
#SBATCH --partition=gpu-common,scavenger-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yl1064@duke.edu
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --exclude=dcc-gehmlab-gpu-ferc-s-z25-18,dcc-carlsonlab-gpu-ferc-s-h36-24,dcc-core-gpu-ferc-s-h36-6,dcc-majoroslab-gpu-ferc-a-r32-34

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python_path="${ECE661_PYTHON:-$HOME/miniforge3/envs/ece661-project4/bin/python}"
export PYTHONNOUSERSITE=1
: "${MODE:?set MODE=linear or MODE=finetune}"

runs=(rotnet_r50_b256_lr0p1_seed0 rotnet_r50_b1024_lr0p4_seed0)
batches=(256 1024)
task=$SLURM_ARRAY_TASK_ID
common=(--seed 0 --subset-seed 0 --num-workers 8 --max-epochs 100 --plateau-patience 10
        --plateau-threshold 0.01 --lr-drops 1 --weight-decay 0 --eval-every 10)
case "$MODE" in
  linear)
    if (( task == 20 )); then
      source_args=(--arch resnet50); batch=256; out=results/r50/linear/random_init
    else
      (( task >= 0 && task < 20 )) || { echo "Unexpected task $task" >&2; exit 1; }
      r=$(( task / 10 )); epoch=$(( (task % 10 + 1) * 100 )); batch=${batches[$r]}
      encoder=$(printf "checkpoints/r50/%s/encoder_epoch%04d.pt" "${runs[$r]}" "$epoch")
      source_args=(--init "$encoder"); out=$(printf "results/r50/linear/%s/epoch%04d" "${runs[$r]}" "$epoch")
    fi
    lr=$("$python_path" -c "print(0.1 * $batch / 256)")   # SimCLR B.6
    args=("${source_args[@]}" --freeze-encoder --fraction 1.0)
    ;;
  finetune)
    (( task >= 0 && task < 8 )) || { echo "Unexpected task $task" >&2; exit 1; }
    fractions=(0.01 0.03 0.05 0.1)
    r=$(( task / 4 )); fraction=${fractions[$(( task % 4 ))]}; batch=${batches[$r]}
    encoder="checkpoints/r50/${runs[$r]}/encoder_epoch1000.pt"
    lr=$("$python_path" -c "print(0.05 * $batch / 256)")  # SimCLR B.5
    args=(--init "$encoder" --fraction "$fraction")
    out="results/r50/finetune/${runs[$r]}/fraction${fraction}"
    ;;
  *) echo "Unknown MODE=$MODE" >&2; exit 1 ;;
esac
echo "Node: $(hostname) | $MODE task $task | batch $batch | lr $lr | out $out | restart ${SLURM_RESTART_COUNT:-0}"

requeue() { echo "Time limit near: requeueing $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; }
trap requeue USR1
"$python_path" -u -m cifar_ssl.downstream "${args[@]}" --batch-size "$batch" --lr "$lr" \
  "${common[@]}" --out-dir "$out" &
wait $!
