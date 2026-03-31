#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --account=ls_polle
#SBATCH --job-name=nerf-train
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH -o logs/nerf_%j.out
#SBATCH -e logs/nerf_%j.err

set -euo pipefail
mkdir -p logs

# Optional modules (if your cluster needs them)
module purge
module load stack/2024-06 gcc/12.2.0
module load cuda/11.8.0
module load eth_proxy

# Correct conda setup for non-interactive shell
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nerfstudio

# Optional sanity check
which python
which ns-train

BASE_DIR="/cluster/scratch/rbollati/models_sparse"

for d in "$BASE_DIR"/nerf_ensemble_*; do
  [ -d "$d" ] || continue
  name="$(basename "$d")"

  if [ "$name" = "nerf_ensemble_1" ]; then
    echo "Skipping $name"
    continue
  fi

  echo "=== Training in $d ==="
  cd "$d"

  ns-train nerfacto \
    --vis tensorboard \
    --experiment-name ensemble_1 \
    --timestamp nerf \
    --steps-per-eval-image 30000 \
    --max-num-iterations 30000 \
    --save-only-latest-checkpoint True \
    --logging.steps-per-log 100 \
    --logging.profiler pytorch \
    colmap \
    --downscale-factor 1 \
    --colmap-path /cluster/scratch/rbollati/dataset/{name of the folder in the outer loop}/sparse/0 \
    --images-path /cluster/scratch/rbollati/dataset/{name of the folder in the outer loop}/images
done