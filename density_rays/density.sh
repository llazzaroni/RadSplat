#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=dslab
#SBATCH --job-name=nerf-train
#SBATCH -o logs/query_%j.out
#SBATCH -e logs/query_%j.er

set +u
source /work/courses/dslab/team20/miniconda3/etc/profile.d/conda.sh
conda activate ns50_ns-upgraded
export PYTHONPATH="$HOME/ds-lab/RadSplat:$PYTHONPATH"
set -u

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [set up env] finished"

cd /work/courses/dslab/team20/rbollati/running_env/experiments/20251121_183303_counter

python ~/ds-lab/RadSplat/density_rays/nerf_query.py
