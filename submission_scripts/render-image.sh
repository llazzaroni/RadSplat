#!/bin/bash
#SBATCH --time=01:00
#SBATCH --account=dslab
#SBATCH --job-name=nerf-train
#SBATCH -o logs/nerf_%j.out
#SBATCH -e logs/nerf_%j.er

###############################################################
# DESCRIPTION: 
# this iscript run the python script used to render a specific
# image
###############################################################

set +u
source /work/courses/dslab/team20/miniconda3/etc/profile.d/conda.sh
conda activate gs50 
set -u

echo "##################### [Job started] #####################"
cd /work/courses/dslab/team20/rbollati/running_env

# python ~/ds-lab/RadSplat/gs_initialization.py 
python ~/ds-lab/RadSplat/test.py 
