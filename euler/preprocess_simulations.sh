#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --job-name=simulations_preprocess
#SBATCH --output=/cluster/home/%u/DrTrack/logs/%x.out                                                                         
#SBATCH --error=/cluster/home/%u/DrTrack/logs/%x.err

module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy

export JAX_PLATFORMS=cpu

$HOME/$1/bin/python3.11 $HOME/DrTrack/src/evaluate_tracking.py \
    data_path=/cluster/scratch/$USER/data/evaluation \
    checkpoint_dir=/cluster/scratch/$USER/checkpoints \
    experiment_name=small_all \
    skip_tracking=true \
    skip_results_generation=true \
    extract_visual_embeddings=droplets_all \
    simulated_image=small_mvt_6000_droplets.csv \
    evaluate=unbalanced_v1 \
    device=cpu

$HOME/$1/bin/python3.11 $HOME/DrTrack/src/evaluate_tracking.py \
    data_path=/cluster/scratch/$USER/data/evaluation \
    checkpoint_dir=/cluster/scratch/$USER/checkpoints \
    experiment_name=medium_all \
    skip_tracking=true \
    skip_results_generation=true \
    extract_visual_embeddings=droplets_all \
    simulated_image=medium_mvt_6000_droplets.csv \
    evaluate=unbalanced_v1 \
    device=cpu

$HOME/$1/bin/python3.11 $HOME/DrTrack/src/evaluate_tracking.py \
    data_path=/cluster/scratch/$USER/data/evaluation \
    checkpoint_dir=/cluster/scratch/$USER/checkpoints \
    experiment_name=large_all \
    skip_tracking=true \
    skip_results_generation=true \
    extract_visual_embeddings=droplets_all \
    simulated_image=large_mvt_6000_droplets.csv \
    evaluate=unbalanced_v1 \
    device=cpu
