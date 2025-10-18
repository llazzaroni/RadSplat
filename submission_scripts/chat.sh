#!/bin/bash
#SBATCH --time=01:00
#SBATCH --account=dslab
#SBATCH --job-name=nerf-train
#SBATCH -o logs/nerf_%j.out
#SBATCH -e logs/nerf_%j.er


echo "Job started: $(date)"

# ---------- GPU info ----------
nvidia-smi || true

# ---------- Paths ----------
DATA_DIR="/work/courses/dslab/team20/data/bicycle"
IMAGES_PATH="$DATA_DIR/images"
COLMAP_PATH="$DATA_DIR/sparse/0"     # change to "$DATA_DIR/sparse" if you don't have the "0" subfolder

# ---------- CUDA (module + env) ----------
module purge
module load cuda/12.8

# Force PyTorch/extension builders to use the CUDA 12.8 toolkit from the module:
export CUDA_HOME="/cluster/data/cuda/12.8.1"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "Using nvcc at: $(which nvcc)"
nvcc --version || true
echo "CUDA_HOME: $CUDA_HOME"

# ---------- Conda env ----------
# Source conda properly for non-interactive shells
conda activate ns50

# ---------- Arch / perf knobs ----------
export TCNN_CUDA_ARCHITECTURES=120
export TORCH_CUDA_ARCH_LIST="12.0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fewer dataloader workers to avoid oversubscription on the node
NUM_WORKERS=2

# ---------- Quick sanity check ----------
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

# ---------- (Optional) ensure CUDA backend for ZipNeRF is present ----------
# If you had to build _cuda_backend yourself, it should already be installed in this env.
# You can uncomment this probe to confirm it's importable:
# python -c "import importlib.util as iu; spec=iu.find_spec('_cuda_backend'); print('_cuda_backend at:', spec.origin if spec else None)"

# ---------- Kick off training ----------
echo "Starting training at: $(date)"
ns-train zipnerf colmap \
  --data "$DATA_DIR" \
  --colmap-path "$COLMAP_PATH" \
  --images-path "$IMAGES_PATH" \
  --pipeline.datamanager.num-workers $NUM_WORKERS
echo "Job finished: $(date)"

