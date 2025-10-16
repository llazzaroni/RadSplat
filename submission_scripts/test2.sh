#!/bin/bash
#SBATCH --time=01:00
#SBATCH --account=dslab
#SBATCH --job-name=nerf-train
#SBATCH -o logs/nerf_%j.out
#SBATCH -e logs/nerf_%j.er

set -euxo pipefail

echo "Job started on $(hostname) at $(date)"

# --- Ensure logs dir exists
mkdir -p logs

# --- (Optional) load modules if your cluster uses environment modules
# module load cuda/12.1
# module load gcc/12.2

# --- Properly initialize conda in non-interactive shells
# source "$HOME/miniconda3/etc/profile.d/conda.sh" || source "$HOME/anaconda3/etc/profile.d/conda.sh"

conda init
echo "Activating conda environment"
conda activate nerfstudio

# --- Sanity checks
which python
python -c "import torch, sys; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'Device count:', torch.cuda.device_count()); sys.exit(0)"
nvidia-smi || true

# --- (One-time) make sure ZipNeRF deps are present in the env
# Comment these if you've already installed them into the env.
pip install --no-input torch_scatter --index-url https://pytorch-geometric.com/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])").html || true
pip install --no-input "git+https://github.com/SuLvXiangXin/zipnerf-pytorch#subdirectory=extensions/cuda" || true
pip install --no-input "git+https://github.com/SuLvXiangXin/zipnerf-pytorch" || true

echo "Navigating into nerfstudio repo (if you need local extras)"
cd /home/rbollati/ds-lab/RadSplat/submodules/nerfstudio

# --- Common training flags
MAX_ITERS=20000                      # tune as needed
OUT_DIR=/work/courses/dslab/team20/outputs/zipnerf
EXP_NAME=bicycle_zipnerf
VIS_BACKEND=tensorboard              # avoids launching the interactive viewer on headless nodes

mkdir -p "$OUT_DIR"

# =========================
# == Choose ONE of these ==
# =========================

# (A) If your $DATA_DIR is ALREADY in nerfstudio processed format (contains transforms.json):
DATA_DIR=/work/courses/dslab/team20/data/bicycle
ns-train zipnerf nerfstudio-data \
    --data "$DATA_DIR" \
    --max-num-iterations $MAX_ITERS \
    --experiment-name "$EXP_NAME" \
    --output-dir "$OUT_DIR" \
    --vis "$VIS_BACKEND"

# (B) If your data is a raw COLMAP project (images/ and sparse/0 inside $DATA_DIR):
# DATA_DIR=/work/courses/dslab/team20/data/bicycle
# ns-train zipnerf \
#     --pipeline.datamanager.dataparser colmap \
#     --data "$DATA_DIR" \
#     --pipeline.datamanager.dataparser.images_path images \
#     --pipeline.datamanager.dataparser.colmap_path sparse/0 \
#     --max-num-iterations $MAX_ITERS \
#     --experiment-name "$EXP_NAME" \
#     --output-dir "$OUT_DIR" \
#     --vis "$VIS_BACKEND"

echo "Training finished at $(date)"
