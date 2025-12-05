#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --account=dslab_jobs
#SBATCH --job-name=nerf-train
#SBATCH -o logs/nerf_%j.out
#SBATCH -e logs/nerf_%j.er

set -euo pipefail

[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
[ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash

set +u
module purge
module load cuda/12.8
module load gcc/12
set -u

set +u
source /work/courses/dslab/team20/miniconda3/etc/profile.d/conda.sh
conda activate gsplat-gpu
set -u

pip uninstall -y fused-ssim || true

pip install --no-build-isolation --no-binary :all: --force-reinstall \
    git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5

RENDER_TRAJ_PATH="ellipse"
DATA_FACTOR=4
STEPS_GS=5000



python ~/ds-lab/RadSplat/save_stats.py default \
    --nerf-init \
    --pt-path "/home/llazzaroni/ds-lab/RadSplat/sub_patches/data/ray_sample.pt" \
    --save-first-ckp \
    --eval-steps -1 \
    --disable-viewer \
    --data-factor "$DATA_FACTOR" \
    --render-traj-path "$RENDER_TRAJ_PATH" \
    --data-dir "/work/courses/dslab/team20/data/mipnerf360/counter/" \
    --result-dir "/home/llazzaroni/ds-lab/RadSplat/sub_patches/data/gsplat_nerf/" \
    --max-steps "$STEPS_GS"