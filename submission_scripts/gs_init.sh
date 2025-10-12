#!/bin/bash   
#SBATCH --job-name=360_nerf_test #Name of the job   
#SBATCH --ntasks=1               #Requesting 1 node (is always 1)
#SBATCH --cpus-per-task=1        #Requesting 1 CPU
#SBATCH --mem-per-cpu=32G        #Requesting 64 Gb memory per core 
#SBATCH --time=3:00:00           #Requesting 12 hours running time 
#SBATCH --output=/cluster/home/rbollati/logs/logs_nerf.log    #Log
#SBATCH --error=/cluster/home/rbollati/logs/logs_nerf_err.log    #Log

##########################################
echo "Starting nerf test"
echo "$(date) start ${SLURM_JOB_ID}"
##########################################

cd ~/ds-lab/RadSpalt

source env/bin/activate

python gs_init_zipnerf.py 

##############################################
##Get a summary of the job 
jeffrun -j ${SLURM_JOB_ID}
##############################################

