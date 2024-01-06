#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --ntasks=4
##SBATCH --gpus=rtx_4090:1
##SBATCH --mem-per-cpu=10G
#SBATCH --job-name=track_droplets
#SBATCH --output=/cluster/home/%u/DrTrack/logs/%x.out                                                                         
#SBATCH --error=/cluster/home/%u/DrTrack/logs/%x.err

#module load gcc/8.2.0 python_gpu/3.11.2

cd ..
python src/evaluate_tracking.py
#export JAX_PLATFORMS=cuda

# $HOME/$1/bin/python3.11 $HOME/DrTrack/src/track_droplets.py \
#     data_path=/cluster/scratch/$USER/data \
#     checkpoint_dir=/cluster/scratch/$USER/checkpoints \
#     track=tau_999 \
#     raw_image="Small mvt 3.nd2" \
#     extract_visual_embeddings=droplets_all \
#     device=cuda \
#     track.relative_epsilon=5e-3 \
#     track.embedding_dist=euclid \
#     track.alpha=0.25 \
