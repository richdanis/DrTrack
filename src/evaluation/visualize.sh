#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=20G
#SBATCH --job-name=dslab_training
#SBATCH --output=/cluster/home/%u/DrTrack/logs/%x.out
#SBATCH --error=/cluster/home/%u/DrTrack/logs/%x.err

module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy tmux r/4.1.3 gsl/2.6 #texlive/live

# need to change to your environment here (mine is just called lab_env)
python3 $HOME/DrTrack/src/evaluation/visualize.py \
    --test_data_path /cluster/scratch/wormaniec/old_dsl_data/cell_datasets/c334_unstimulated_crop4x4.npy \
    --validation_batch_size 32 \
    --embed_dim 10 \
    --topk_accuracy 1 5 \
    --checkpoint_path /cluster/scratch/wormaniec/model_checkpoints/2023-11-20_16-51-16_embeddings_model.pt
