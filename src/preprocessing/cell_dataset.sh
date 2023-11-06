#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=10G
#SBATCH --job-name=cell_dataset
#SBATCH --output=/cluster/home/%u/DrTrack/logs/%x.out                                                                         
#SBATCH --error=/cluster/home/%u/DrTrack/logs/%x.err

module load gcc/8.2.0 python_gpu/3.11.2

python3 $HOME/DrTrack/src/preprocessing/cell_dataset.py --fname "$1" 
