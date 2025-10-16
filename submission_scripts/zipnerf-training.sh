#!/bin/bash
#SBATCH --time=01:00
#SBATCH --account=dslab
#SBATCH --job-name=nerf-train
#SBATCH -o logs/nerf_%j.out
#SBATCH -e logs/nerf_%j.er

echo "Job started"

nvidia-smi

DATA_DIR=/work/courses/dslab/team20/data/bicycle


echo "activating conda environmern"
conda activate nerfstudio 

echo "navigating into the code directory"
cd /home/rbollati/ds-lab/RadSplat/submodules/nerfstudio

# echo "running zipnerf cli tool"
# ns-install-cli

echo "starting training"

ns-train zipnerf --data "$DATA_DIR"
  # --pipeline.datamanager.dataparser colmap \
  # --data "$DATA_DIR" \
  # --pipeline.datamanager.dataparser.images_path images \
  # --pipeline.datamanager.dataparser.colmap_path sparse/0

