#!/bin/bash
#SBATCH --time=02:30:00
#SBATCH --account=dslab_jobs
#SBATCH --job-name=nerf-train
#SBATCH -o logs/nerf_%j.out
#SBATCH -e logs/nerf_%j.er

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

scenes="bicycle flowers"

export RUNNING_DIR="/work/courses/dslab/team20/rbollati/running_env"
export BASE_DATA_DIR="/work/courses/dslab/team20/data/mipnerf360"

for scenename in $scenes;do

  echo "---------------------- [Scene: ${scenename}] ----------------------"

  export SCENE=$scenename

  ## Folders and output names
  export EXPERIMENT_NAME="$(date '+%Y%m%d_%H%M%S')_${SCENE}_SFM"
  export EXPERIMENT_DIR="${RUNNING_DIR}/experiments/${EXPERIMENT_NAME}"
  export DATA_DIR="${BASE_DATA_DIR}/${SCENE}"
  export NERF_MODEL="${EXPERIMENT_DIR}/outputs/${EXPERIMENT_NAME}/nerfacto/${EXPERIMENT_NAME}/config.yml"
  export STEPS_GS=30000

  echo "##################### [Job started] #####################"
  mkdir "${EXPERIMENT_DIR}"
  cd "${EXPERIMENT_DIR}"

  echo "##################### [STARTING GSPLAT] #####################"


  RENDER_TRAJ_PATH="ellipse"
  DATA_FACTOR=4

  python ~/ds-lab/RadSplat/save_stats.py default \
    --no-nerf-init \
    --save-first-ckp \
    --eval-steps -1 \
    --disable-viewer \
    --data-factor "$DATA_FACTOR" \
    --render-traj-path "$RENDER_TRAJ_PATH" \
    --data-dir "$DATA_DIR/" \
    --result-dir "$EXPERIMENT_DIR/" \
    --max-steps "$STEPS_GS"

done
