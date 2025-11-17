#!/bin/bash
#SBATCH --time=01:00
#SBATCH --account=dslab
#SBATCH --job-name=nerf-train
#SBATCH -o logs/init_%j.out
#SBATCH -e logs/init_%j.er

set -euo pipefail

module purge
module load cuda/12.8
module load gcc/12

source /work/courses/dslab/team20/miniconda3/etc/profile.d/conda.sh
conda activate gsplat-gpu


export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="12.0"

pip uninstall -y fused-ssim || true

pip install --no-build-isolation --no-binary :all: --force-reinstall \
  git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5

SCENE_DIR="/work/courses/dslab/team20/data/bicycle"
RESULT_DIR="/work/courses/dslab/team20/results_gsplat"
RENDER_TRAJ_PATH="ellipse"
DATA_FACTOR=4

REPO_DIR="/home/llazzaroni/RadSplat/submodules/gsplat"
cd "$REPO_DIR"

srun python examples/simple_trainer.py default \
  --eval_steps -1 \
  --disable_viewer \
  --data_factor "$DATA_FACTOR" \
  --render_traj_path "$RENDER_TRAJ_PATH" \
  --data_dir "$SCENE_DIR/" \
  --result_dir "$RESULT_DIR/"