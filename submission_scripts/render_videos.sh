#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=dslab
#SBATCH --job-name=nerf-train
#SBATCH -o logs/video_%j.out
#SBATCH -e logs/video_%j.er

###############################################################
# DESCRIPTION: 
# FULL pipeline to run nerfacto and sample points 
###############################################################


[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
[ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash

set +u
module purge
module load cuda/12.8
module load gcc/12
set -u

set +u
source /work/courses/dslab/team20/miniconda3/etc/profile.d/conda.sh
conda activate gsplat-gpu
set -u

pip uninstall -y fused-ssim || true

pip install --no-build-isolation --no-binary :all: --force-reinstall \
git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5

echo "------------------ ENVIRONMENT BUILT --------------------"

RENDER_TRAJ_PATH="ellipse"
DATA_FACTOR=4

export RUNNING_DIR="/work/courses/dslab/team20/rbollati/running_env"
export BASE_DATA_DIR="/work/courses/dslab/team20/data/mipnerf360"
export EXPERIMENTS_ROOT="${RUNNING_DIR}/experiments"

for EXPERIMENT_DIR in "${EXPERIMENTS_ROOT}"/*; do

  echo "---------------------- [Experiment: ${EXPERIMENT_DIR}}] ----------------------"

  if [ -d "${EXPERIMENT_DIR}/videos" ]; then
    echo "Skipping $(basename "$EXPERIMENT_DIR"): 'videos' folder already exists."
    continue
  fi

  export DATA_DIR="${BASE_DATA_DIR}/bonsai"
  export CKPT="${EXPERIMENT_DIR}/ckpts/ckpt_0_rank0.pt"

  echo "##################### [Job started] #####################"
  cd "${EXPERIMENT_DIR}"

  echo "##################### [STARTING GSPLAT] #####################"

  python ~/ds-lab/RadSplat/save_stats.py default \
    --no-nerf-init \
    --save-first-ckp \
    --eval-steps -1 \
    --disable-viewer \
    --data-factor "$DATA_FACTOR" \
    --render-traj-path "$RENDER_TRAJ_PATH" \
    --data-dir "$DATA_DIR/" \
    --result-dir "$EXPERIMENT_DIR/" \
    --ckpt "$CKPT"

done
