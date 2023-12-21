#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=20G
#SBATCH --job-name=test
#SBATCH --output=/cluster/home/%u/DrTrack/logs/%x.out                                                                         
#SBATCH --error=/cluster/home/%u/DrTrack/logs/%x.err

module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy

export JAX_PLATFORMS=cuda

$HOME/$1/bin/python3.11 $HOME/DrTrack/src/evaluate_tracking.py \
    data_path=/cluster/scratch/$USER/data/evaluation \
    checkpoint_dir=/cluster/scratch/$USER/checkpoints \
    experiment_name=small_all \
    extract_visual_embeddings=droplets_all \
    simulated_image=small_mvt_20000_droplets.csv \
    skip_preprocessing=true \
    skip_visual_embedding_extraction=true \
    skip_calibration_plot=true \
    skip_tracking=false \
    skip_results_generation=false \
    evaluate=unbalanced_v1 \
    track=tau_9999 \
    device=cuda \
    wandb=true \
    evaluate.auroc=false

$HOME/$1/bin/python3.11 $HOME/DrTrack/src/evaluate_tracking.py \
    data_path=/cluster/scratch/$USER/data/evaluation \
    checkpoint_dir=/cluster/scratch/$USER/checkpoints \
    experiment_name=medium_all \
    extract_visual_embeddings=droplets_all \
    simulated_image=medium_mvt_20000_droplets.csv \
    skip_preprocessing=true \
    skip_visual_embedding_extraction=true \
    skip_calibration_plot=true \
    skip_tracking=false \
    skip_results_generation=false \
    evaluate=unbalanced_v1 \
    track=tau_9999 \
    device=cuda \
    wandb=true \
    evaluate.auroc=false

$HOME/$1/bin/python3.11 $HOME/DrTrack/src/evaluate_tracking.py \
    data_path=/cluster/scratch/$USER/data/evaluation \
    checkpoint_dir=/cluster/scratch/$USER/checkpoints \
    experiment_name=large_all \
    extract_visual_embeddings=droplets_all \
    simulated_image=large_mvt_20000_droplets.csv \
    skip_preprocessing=true \
    skip_visual_embedding_extraction=true \
    skip_calibration_plot=true \
    skip_tracking=false \
    skip_results_generation=false \
    evaluate=unbalanced_v1 \
    track=tau_9999 \
    device=cuda \
    wandb=true \
    evaluate.auroc=false
