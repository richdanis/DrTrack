#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=10G
#SBATCH --job-name=merge
#SBATCH --output=/cluster/home/%u/DrTrack/logs/%x.out                                                                         
#SBATCH --error=/cluster/home/%u/DrTrack/logs/%x.err

module load gcc/8.2.0 python_gpu/3.11.2

# could not extract anything from small_5
python3 $HOME/DrTrack/src/preprocessing/merge.py --fnames \
    "small_1" \
    "small_2" \
    "small_3" \
    "small_4" \
    "small_6" \
    "small_7" \
    "small_8"
