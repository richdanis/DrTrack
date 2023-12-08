#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=20G
#SBATCH --job-name=track_droplets
#SBATCH --output=/cluster/home/%u/DrTrack/logs/%x.out                                                                         
#SBATCH --error=/cluster/home/%u/DrTrack/logs/%x.err

module load gcc/8.2.0 python_gpu/3.11.2

export JAX_PLATFORMS=cuda,cpu

$HOME/$1/bin/python3.11 $HOME/DrTrack/src/track_droplets.py