#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=track_droplets
##SBATCH --output=/cluster/home/%u/DrTrack/logs/%x.out                                                        
##SBATCH --error=/cluster/home/%u/DrTrack/logs/%x.err
#SBATCH --output=logs/%x.out                                                        
#SBATCH --error=logs/%x.err

module load gcc/8.2.0 python_gpu/3.11.2

#export JAX_PLATFORMS=cuda

cd ..
#$HOME/dr_track/bin/python3.11 $HOME/DrTrack/src/track_droplets.py \
#data_path=/cluster/scratch/$USER/data \
#checkpoint_dir=/cluster/scratch/$USER/checkpoints \

$HOME/dr_track/bin/python3.11 src/track_droplets.py \
    experiment_name="large_mvt_4" \
    skip_preprocessing=true \
    skip_droplet_detection=true \
    skip_droplet_patch_extraction=true \
    skip_visual_embedding_extraction=false \
    skip_tracking=false \
    skip_results_generation=false \
    track=medium_20000_best \
    raw_image="Large mvt 4.nd2" \
    extract_visual_embeddings=droplets_all \
    generate_results.calibrate_probabilities=true \
    generate_results.calibration_model_name="large_mvt_6000.pkl" \
    device=cpu \
    #checkpoint_dir=/cluster/scratch/$USER/checkpoints \
    