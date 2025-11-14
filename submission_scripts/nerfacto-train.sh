#!/bin/bash
#SBATCH --time=01:00
#SBATCH --account=dslab
#SBATCH --job-name=nerf-train
#SBATCH -o logs/nerf_%j.out
#SBATCH -e logs/nerf_%j.er

echo "##################### [Job started] #####################"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [set up env] started"

echo "[set up env] navigating to folder"
cd /work/courses/dslab/team20/rbollati/running_env

echo "[set up env] activate conda environment"
conda init
conda activate ns50 

module load cuda/12.8

# echo "[set up env] CUDA sanity check"
# python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda)"
#


# echo "[Donwload data] download nerfstudio data"
# ns-download-data nerfstudio --capture-name=poster

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Training] started"
ns-train nerfacto \
  --vis tensorboard \
  --experiment-name "nerfacto-poster-3k-steps-allcheck" \
  --steps-per-eval-image 100 \
  --max-num-iterations 3000 \
  --steps-per-save 300 \
  --save-only-latest-checkpoint False \
  --logging.steps-per-log 200 \
  --data data/nerfstudio/poster

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Training] finished"

# ns-train nerfacto  \
#   --vis viewer+tensorboard \
#   --steps-per-eval-image 10 \
#   --logging.steps-per-log 1 \
#   --logging.local-writer.enable True \
#   colmap \
#   --colmap-path /work/courses/dslab/team20/data/bicycle/sparse/0 \
#   --images-path /work/courses/dslab/team20/data/bicycle/images
