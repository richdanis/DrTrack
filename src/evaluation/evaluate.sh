#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=10G
#SBATCH --job-name=evaluation
#SBATCH --output=/cluster/home/%u/DrTrack/logs/%x.out
#SBATCH --error=/cluster/home/%u/DrTrack/logs/%x.err

module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy tmux r/4.1.3 gsl/2.6 #texlive/live

# need to change to your environment here (mine is just called lab_env)
$HOME/lab_env/bin/python3.11 $HOME/DrTrack/src/evaluation/evaluate.py \
    --test_data_path /cluster/scratch/$USER/data/local_datasets/validation \
    --validation_batch_size 32 \
    --embed_dim 20 \
    --checkpoint_path /cluster/scratch/$USER/checkpoints/2023-12-04_10-34-14_dim_20.pth \
    --use_dapi