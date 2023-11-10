#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
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
    --epochs 1 \
    --lr 1e-3 \
    --device cuda \
    --topk_accuracy 1 5 \
    --wandb True


