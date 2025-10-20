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

module load cuda/12.8

echo "[set up env] activate conda environment"
source /work/courses/dslab/team20/miniconda3/etc/profile.d/conda.sh
conda activate ns50 

echo "[set up env] CUDA sanity check"
python -c "import torch; print('torch:', torch._version_); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda)"


echo "[Starting] script"
python ~/RadSplat/nerf_query.py
