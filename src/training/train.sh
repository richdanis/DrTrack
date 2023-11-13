#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10g
#SBATCH --mem-per-cpu=20G
#SBATCH --job-name=dslab_training
#SBATCH --output=/cluster/home/%u/DrTrack/logs/%x.out                                                                         
#SBATCH --error=/cluster/home/%u/DrTrack/logs/%x.err

module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy

# need to change to your environment here (mine is just called lab_env)
$HOME/lab_env/bin/python3.11 $HOME/DrTrack/src/training/train.py \
    --data_path /cluster/scratch/$USER/data/ \
    --model efficientnet-b0 \
    --batch_size 32 \
    --epochs 60 \
    --lr 1e-3 \
    --device cuda \
    --topk_accuracy 1 5 \
    --wandb \
    --checkpoint_path /cluster/home/rdanis/DrTrack/data/ \
    --embed_dim 128 \
    --samples_per_epoch 500 \
    --validation_batch_size 128 \


