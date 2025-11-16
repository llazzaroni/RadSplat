#!/bin/bash
#SBATCH --time=01:00
#SBATCH --account=dslab
#SBATCH --job-name=nerf-train
#SBATCH -o logs/nerf_%j.out
#SBATCH -e logs/nerf_%j.er

###############################################################
# DESCRIPTION: 
# FULL pipeline to run nerfacto and sample points 
###############################################################

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [set up env] started"

set +u
source /work/courses/dslab/team20/miniconda3/etc/profile.d/conda.sh
conda activate ns50 
set -u

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [set up env] finished"

export RUNNING_DIR="/work/courses/dslab/team20/rbollati/running_env"
export BASE_DATA_DIR="/work/courses/dslab/team20/data/mipnerf360"
export SCENE="flowers"

export EXPERIMENT_NAME="$(date '+%Y%m%d_%H%M%S')"
export EXPERIMENT_DIR="${RUNNING_DIR}/experiments/${EXPERIMENT_NAME}"

## PARAMETERS NERF
export NERF_MAX_NUM_ITERATIONS=500
export DATA_DIR="${BASE_DATA_DIR}/${SCENE}"
export NERF_MODEL="${EXPERIMENT_DIR}/outputs/${EXPERIMENT_NAME}/nerfacto/${EXPERIMENT_NAME}/config.yml"

export POSITION_TENSOR_OUTPUT_NAME="${EXPERIMENT_DIR}/ray_sample.pt"
export RAY_SAMPLING_STRATEGY="canny"

mkdir "${EXPERIMENT_DIR}"

echo "##################### [Job started] #####################"
cd "${EXPERIMENT_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Training-nerfacto] started"

ns-train nerfacto \
  --vis tensorboard \
  --experiment-name "${EXPERIMENT_NAME}" \
  --timestamp "${EXPERIMENT_NAME}" \
  --steps-per-eval-image $NERF_MAX_NUM_ITERATIONS \
  --max-num-iterations $NERF_MAX_NUM_ITERATIONS \
  --save-only-latest-checkpoint True \
  --logging.steps-per-log 100 \
  --logging.profiler "pytorch" \
  colmap \
  --colmap-path "${DATA_DIR}/sparse/0" \
  --images-path "${DATA_DIR}/images"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Training-nerfacto] finished"

START_SAMPLING=$(($(date +%s)*1000))
python ~/ds-lab/RadSplat/nerf_step.py --nerf-folder $NERF_MODEL --output-name $POSITION_TENSOR_OUTPUT_NAME --sampling-size 500000 --ray-sampling-strategy $RAY_SAMPLING_STRATEGY 
END_SAMPLING=$(($(date +%s)*1000))

echo "##################### [FINISHED NERF] #####################"
echo "##################### [STARTING GSPLAT] #####################"

set +u
conda deactivate
set -u

set -euo pipefail

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

RENDER_TRAJ_PATH="ellipse"
DATA_FACTOR=4

srun python save_stats.py default \
  --nerf_init True \
  --pt_path "$POSITION_TENSOR_OUTPUT_NAME" \
  --save_first_ckp True \
  --eval_steps -1 \
  --disable_viewer \
  --data_factor "$DATA_FACTOR" \
  --render_traj_path "$RENDER_TRAJ_PATH" \
  --data_dir "$DATA_DIR/" \
  --result_dir "$EXPERIMENT_DIR/"
