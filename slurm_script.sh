#!/bin/bash

#SBATCH --job-name=graphvae_torch
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -J graph_vae_torch
#SBATCH -p gpu
#SBATCH --qos gpu
#SBATCH --gres gpu:1
#SBATCH --mem=32G 
#SBATCH --output=./Data/log.txt

source /share/apps/modulefiles/conda_init.sh
conda activate torch-gpu

python3 hpc.py