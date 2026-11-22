#!/bin/bash
# Submit a ResNet-50 job to any free GPU with at least 24 GB of memory.
#   bash scripts/slurm/r50_sbatch.sh [sbatch options] scripts/slurm/<job>.sbatch
# Builds --exclude from the live node list: every GPU node in the two partitions
# whose GPU type is not on the 24 GB+ allowlist, plus the four nodes the project
# always avoids. Passes all other arguments to sbatch unchanged.
set -euo pipefail
always="dcc-gehmlab-gpu-ferc-s-z25-18,dcc-carlsonlab-gpu-ferc-s-h36-24,dcc-core-gpu-ferc-s-h36-6,dcc-majoroslab-gpu-ferc-a-r32-34"
allowed='gpu:(5000_ada|a5000|a6000|6000_ada|6000_ada_generation|a100|h100|h200|l40|l40s)[:(]'
small=$(sinfo -p gpu-common,scavenger-gpu -h -N -o "%N %G" | awk -v ok="$allowed" '$2 !~ ok {print $1}' | sort -u | paste -sd, -)
exclude="$always${small:+,$small}"
echo "Excluding $(echo "$exclude" | tr ',' '\n' | sort -u | wc -l) nodes without a 24 GB+ GPU or on the avoid list" >&2
exec sbatch --exclude="$exclude" "$@"
