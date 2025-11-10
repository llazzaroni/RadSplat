#!/bin/bash
#SBATCH --time=01:00
#SBATCH --account=dslab
#SBATCH --job-name=nerf-train
#SBATCH -o logs/nerf_%j.out
#SBATCH -e logs/nerf_%j.er

###############################################################
# DESCRIPTION: 
# This script sample points using the gs_initialization.py
# script
###############################################################

set +u
source /work/courses/dslab/team20/miniconda3/etc/profile.d/conda.sh
conda activate gs50 
set -u


export RUNNING_DIR="/work/courses/dslab/team20/rbollati/running_env"
export NERF_MODEL="$RUNNING_DIR/outputs/poster/nerfacto/2025-10-18_013814/config.yml"
export OUTPUT_NAME="$RUNNING_DIR/ray_samples/sobel"

echo "##################### [Job started] #####################"
cd 

python ~/ds-lab/RadSplat/gs_initialization.py 

echo "##################### [TERMINATED] #####################"
