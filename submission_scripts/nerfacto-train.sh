#!/bin/bash
#SBATCH --time=01:00
#SBATCH --account=dslab
#SBATCH --job-name=nerf-train
#SBATCH -o logs/nerf_%j.out
#SBATCH -e logs/nerf_%j.er

echo "##################### [Job started] #####################"
echo "[set up env] navigating to folder"
cd /work/courses/dslab/team20/rbollati/running_env

echo "[set up env] nvidia-smi"
nvidia-smi

# DATA_DIR=/work/courses/dslab/team20/data/bicycle

echo "[set up env] activate conda environment"
conda init
conda activate ns50 

module load cuda/12.8

echo "[set up env] CUDA sanity check"
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda)"


echo "[Donwload data] download nerfstudio data"
# ns-download-data nerfstudio --capture-name=poster


echo "[Training] training model"
ns-train nerfacto \
  --vis viewer+tensorboard \
  --steps-per-eval-image 100 \
  --logging.steps-per-log 10 \
  --logging.local-writer.enable True \
  --logging.local-writer.max-log-size=0 \
  --data data/nerfstudio/poster

# ns-train nerfacto  \
#   --vis viewer+tensorboard \
#   --steps-per-eval-image 10 \
#   --logging.steps-per-log 1 \
#   --logging.local-writer.enable True \
#   colmap \
#   --colmap-path /work/courses/dslab/team20/data/bicycle/sparse/0 \
#   --images-path /work/courses/dslab/team20/data/bicycle/images
