#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --gpus=a100_80gb:1
#SBATCH --mem-per-cpu=30G
#SBATCH --output=/cluster/home/%u/DrTrack/logs/%x.out                                                                         
#SBATCH --error=/cluster/home/%u/DrTrack/logs/%x.err

export JAX_PLATFORMS=cuda

module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy

# pass sweep id
wandb agent $1