#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --ntasks=1
##SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=20G
#SBATCH --job-name=evaluate_tracking_small
##SBATCH --output=/cluster/home/%u/DrTrack/logs/%x.out                                                        
##SBATCH --error=/cluster/home/%u/DrTrack/logs/%x.err
#SBATCH --output=logs/%x.out                                                        
#SBATCH --error=logs/%x.err

#module load gcc/8.2.0 python_gpu/3.11.2

cd ..
#data_path=/cluster/scratch/$USER/evaluation \
#$HOME/dr_track/bin/python3.11 $HOME/DrTrack/src/evaluate_tracking.py \

$HOME/dr_track/bin/python3.11 src/evaluate_tracking.py \
    experiment_name="small_mvt_20000" \
    simulated_image="small_mvt_20000_droplets.csv" \
    skip_preprocessing=true \
    skip_visual_embedding_extraction=true \
    skip_tracking=true \
    skip_trajectory_generation=false \
    skip_scoring=false \
    skip_calibration=false \
    track=small_mvt_1 \
    extract_visual_embeddings=droplets_all \
    tqdm_disable=true \
    device=cpu \
    generate_results.calibrate_probabilities=false \
    generate_results.calibration_model_name="large_mvt_6000.pkl" \
 
