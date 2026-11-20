#!/bin/bash
# Part 2 uses the verified RotNet anchor and shared label subsets.
# FRACTIONS (default "0.01 0.1 1.0"; n values) sets the label fractions.
# Tasks 0..n-1 fine-tune, n..2n-1 train from scratch, 2n..3n-1 fit frozen probes.
# For other fractions, set --array=0-$((3n-1))%1 to match.
# See docs/slurm.md for submission and retry instructions.
#SBATCH --job-name=p4-rotnet-part2
#SBATCH --partition=gpu-common,scavenger-gpu
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yl1064@duke.edu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-8%1
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --exclude=dcc-gehmlab-gpu-ferc-s-z25-18,dcc-carlsonlab-gpu-ferc-s-h36-24,dcc-core-gpu-ferc-s-h36-6,dcc-majoroslab-gpu-ferc-a-r32-34

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python_path="${ECE661_PYTHON:-$HOME/miniforge3/envs/ece661-project4/bin/python}"
test -x "$python_path"
export PYTHONNOUSERSITE=1

ENCODER=checkpoints/figure_b7/rotnet_r20_b1024_lr0p1_seed0/encoder_epoch1000.pt
sha256sum -c --quiet <<'EOF'
4104f40c7ee9ef61805fe3550c864a0f18675013c3a2c01b243423be81386a0b  checkpoints/figure_b7/rotnet_r20_b1024_lr0p1_seed0/encoder_epoch1000.pt
5bf2685daa8e395147239d28810fd5799dc846d5d28e2a539f454d2c17729879  subsets/cifar10_1pct_seed0.json
e3db3a5dd1aad62b23c6d6f77b53208b7e93915556c1c1cd6d994bc817a5c6be  subsets/cifar10_10pct_seed0.json
6a963b76b3730dcc426a316e7a115a4333e930144105643cbe78dbc2b50f6b07  subsets/cifar10_5pct_seed0.json
7c00cf4931af2756c19911ecdebcf7dc0caad130ef11930af874b943a9610dfb  subsets/cifar10_8pct_seed0.json
faac99dec20ea83ee02c3147a10e4b0d93a17f21624082227a0e70e79ddbf105  subsets/cifar10_20pct_seed0.json
EOF

echo "Node: $(hostname) | array task: ${SLURM_ARRAY_TASK_ID}"
"$python_path" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0))'

read -r -a fractions <<< "${FRACTIONS:-0.01 0.1 1.0}"
n=${#fractions[@]}
task=$SLURM_ARRAY_TASK_ID
(( task >= 0 && task < 3 * n )) || { echo "Unexpected array task: $task (n=$n)" >&2; exit 1; }
fraction=${fractions[$(( task % n ))]}
group=$(( task / n ))   # 0 fine-tune, 1 from scratch, 2 frozen probe
echo "fractions: ${fractions[*]} | group $group | fraction $fraction"
if (( group == 2 )); then
  exec "$python_path" -u -m cifar_ssl.linear_eval --checkpoint "$ENCODER" --fraction "$fraction" \
    --stage 3 --pool 1 --epochs 100 --seed 0 --subset-seed 0 --num-workers 4
fi
classifier_args=(
  --fraction "$fraction" --epochs 100 --batch-size 128 --lr 0.1 --weight-decay 5e-4
  --seed 0 --subset-seed 0 --num-workers 4 --min-epoch-size 5000 --eval-every 10
)
method=supervised
if (( group == 0 )); then
  method=rotnet_finetune
  classifier_args+=(--init "$ENCODER")
fi

mkdir -p checkpoints/part2
"$python_path" -u -m cifar_ssl.train_classifier "${classifier_args[@]}" \
  --save "checkpoints/part2/${method}_fraction${fraction}_seed0.pt"
