#!/bin/bash
#SBATCH --time=01:00
#SBATCH --account=dslab
#SBATCH --job-name=nerf-train
#SBATCH -o logs/nerf_%j.out
#SBATCH -e logs/nerf_%j.er

module load cuda/12.8
export CUDA_HOME=/cluster/data/cuda/12.8.1
export CUDA_PATH=$CUDA_HOME
export CUDACXX=$CUDA_HOME/bin/nvcc

# 2) Put it first on PATH & LD_LIBRARY_PATH.
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# 3) Verify PyTorch sees this path now:
python - << 'PY'
import torch.utils.cpp_extension as ce
print("CUDA_HOME seen by PyTorch:", ce.CUDA_HOME)
PY
