#!/bin/bash
#SBATCH --time=01:00
#SBATCH --account=dslab
#SBATCH --job-name=nerf-train
#SBATCH -o logs/nerf_%j.out
#SBATCH -e logs/nerf_%j.er

echo "##################### [Job started] #####################"
cd /work/courses/dslab/team20/rbollati/running_env

nvidia-smi

DATA_DIR=/work/courses/dslab/team20/data/bicycle
export TCNN_CUDA_ARCHITECTURES=120

echo "##################### [activating conda environmern] #####################"
conda init
conda activate ns50 

# echo "running zipnerf cli tool"
# ns-install-cli

echo "load cuda module"
module load cuda/12.8

echo "load gcc"
module load gcc/11

export CUDA_HOME="/cluster/data/cuda/12.8.1"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "Using nvcc at: $(which nvcc)"
nvcc --version || true
echo "CUDA_HOME: $CUDA_HOME"

echo "sanity check"

python - << 'PY'
import torch, os
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0), "CC:", torch.cuda.get_device_capability())
print("TCNN_CUDA_ARCHITECTURES:", os.getenv("TCNN_CUDA_ARCHITECTURES"))
try:
    import tinycudann as tcnn
    print("tinycudann import OK")
except Exception as e:
    print("tinycudann import FAILED:", e)
PY

echo "build cuda backend"

export TCNN_CUDA_ARCHITECTURES=120
export TORCH_CUDA_ARCH_LIST="12.0"

pip install -v --no-build-isolation --no-cache-dir \
  "git+https://github.com/SuLvXiangXin/zipnerf-pytorch#subdirectory=extensions/cuda"

pip install -v --no-cache-dir "git+https://github.com/SuLvXiangXin/zipnerf-pytorch"

echo "sanity check"

python - << 'PY'
import importlib.util
spec = importlib.util.find_spec("_cuda_backend")
print("_cuda_backend found at:", spec.origin if spec else None)
PY

echo "starting training"

echo "navigating into the code directory"
cd /home/rbollati/ds-lab/RadSplat/submodules/nerfstudio

ns-train zipnerf colmap --data "$DATA_DIR" \
  --colmap-path /work/courses/dslab/team20/data/bicycle/sparse/0 \
  --images-path /work/courses/dslab/team20/data/bicycle/images
  # --pipeline.datamanager.dataparser colmap \
  # --pipeline.datamanager.dataparser.images_path images \
  # --pipeline.datamanager.dataparser.colmap_path sparse/0
