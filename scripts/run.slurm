#!/bin/bash

#SBATCH --job-name=perov_5          # Name of the job
#SBATCH --nodes=1                      # Request one node
#SBATCH --ntasks-per-node=32           # Number of tasks per node, here specifying 32 cores
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --time=96:00:00                # Job runtime of 98 hours
#SBATCH --nodelist=dscog034          # Specify the node name constraint
#SBATCH --output=my_gpu_job_output_%j.txt  # Standard output and error log


source ~/.bashrc

source ~/conda_init.sh

conda activate latentexp

python latentexp/run.py data=mp_20 expname=mp_20