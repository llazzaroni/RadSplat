#!/bin/bash
#SBATCH --time=01:00
#SBATCH --account=dslab
#SBATCH --job-name=nerf-train
#SBATCH -o logs/eval_%j.out
#SBATCH -e logs/eval_%j.er

set -euo pipefail

module purge
module load cuda/12.8
module load gcc/12

set +u
source /work/courses/dslab/team20/miniconda3/etc/profile.d/conda.sh
conda activate gsplat-gpu
set -u

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="12.0"

pip uninstall -y fused-ssim || true

pip install --no-build-isolation --no-binary :all: --force-reinstall \
  git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5

python - << 'PY'
import torch, numpy, cv2, fused_ssim
print("torch:", torch.__version__, "cuda rt:", torch.version.cuda, "gpu:", torch.cuda.is_available())
if torch.cuda.is_available(): print("sm:", ".".join(map(str, torch.cuda.get_device_capability())))
print("numpy:", numpy.__version__)
print("cv2:", cv2.__version__, "has calib3d:", hasattr(cv2,"getOptimalNewCameraMatrix"))
print("fused_ssim import OK")
PY

SCENE_DIR="/work/courses/dslab/team20/data/poster"
RESULT_DIR="/work/courses/dslab/team20/results_gsplat_poster/edge_sampler_no_smoothing"
RENDER_TRAJ_PATH="ellipse"
DATA_FACTOR=4

REPO_DIR="/home/rbollati/ds-lab/RadSplat"
cd "$REPO_DIR"

echo "##################### [Checkpoint created] #####################"
srun python gs_splat_ckp_0.py default \
  --eval_steps -1 \
  --disable_viewer \
  --data_factor "$DATA_FACTOR" \
  --render_traj_path "$RENDER_TRAJ_PATH" \
  --data_dir "$SCENE_DIR/" \
  --result_dir "$RESULT_DIR/" \
  --ckpt /work/courses/dslab/team20/results_gsplat_poster/edge_sampler_no_smoothing/ckpts/ckpt_0_rank0.pt
