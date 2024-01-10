#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --ntasks=4
##SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=evaluate_tracking
#SBATCH --output=/cluster/home/%u/DrTrack/logs/%x.out                                                                         
#SBATCH --error=/cluster/home/%u/DrTrack/logs/%x.err

#module load gcc/8.2.0 python_gpu/3.11.2

cd ..
# python src/evaluate_tracking.py \
$HOME/dr_track/bin/python3.11 $HOME/DrTrack/src/evaluate_tracking.py \
    experiment_name="medium_mvt_6000" \
    data_path=/cluster/scratch/$USER/evaluation \
    calibration_model_dir=/cluster/home/%u/DrTrack/calibration_models \
    simulated_image="medium_mvt_6000_droplets.csv" \
    skip_preprocessing=true \
    skip_visual_embedding_extraction=true \
    skip_tracking=true \
    skip_trajectory_generation=false \
    skip_scoring=false \
    skip_calibration=false \
    track=medium_20000_best \
    extract_visual_embeddings=droplets_all \
    device=cpu \
    generate_results.calibrate_probabilities=false \
    generate_results.calibration_model_name="large_mvt_6000.pkl" \
 
