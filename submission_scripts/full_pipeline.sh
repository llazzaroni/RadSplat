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

export OUTPUT_NAME="${EXPERIMENT_DIR}/ray_sample.pt"
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
   colmap \
   --colmap-path "${DATA_DIR}/sparse/0" \
   --images-path "${DATA_DIR}/images"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Training-nerfacto] finished"

python ~/ds-lab/RadSplat/nerf_query.py --nerf-folder $NERF_MODEL --output-name $OUTPUT_NAME --ray-sampling-strategy $RAY_SAMPLING_STRATEGY 

echo "##################### [TERMINATED] #####################"
