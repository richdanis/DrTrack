#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --ntasks=4
##SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=track_droplets
#SBATCH --output=/cluster/home/%u/DrTrack/logs/%x.out                                                                         
#SBATCH --error=/cluster/home/%u/DrTrack/logs/%x.err

#module load gcc/8.2.0 python_gpu/3.11.2

cd ..
python src/evaluate_tracking.py \
    experiment_name="medium_mvt_6000" \
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
    generate_results.calibrate_probabilities=true \
    generate_results.calibration_model_name="small_mvt_6000.pkl" \
 
