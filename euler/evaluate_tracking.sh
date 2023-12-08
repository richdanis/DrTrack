#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --job-name=evaluate_tracking
#SBATCH --output=/cluster/home/%u/Pipeline/logs/%x.out                                                                         
#SBATCH --error=/cluster/home/%u/Pipeline/logs/%x.err

module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy

export JAX_PLATFORMS=cpu

$HOME/$1/bin/python3.11 $HOME/Pipeline/src/evaluate_tracking.py \
    wandb=true data_path=/cluster/scratch/$USER/data/evaluation \
    checkpoint_dir=/cluster/scratch/$USER/checkpoints \
    experiment_name=ululu