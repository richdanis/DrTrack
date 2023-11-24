import sys

import argparse
import logging
import os

import pandas as pd
import wandb
import datetime
import time
import torch
from info_nce import InfoNCE
import numpy as np

sys.path.append('src/')
sys.path.append('src/training/')
from training.models.efficientnet import EfficientNet
from training.dataset import CellDataset
from training.train import evaluate
from setup import get_args, load_model


def main():
    args = get_args()
    wandb.init(
        project="DrTrack",
        entity="dslab-23",
        config=vars(args),
        dir="logs"
    )

    model = load_model(args)
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(args.device))
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Model path: {args.checkpoint_path}")
    logging.info(f"Model loaded.")

    eval_dataset = CellDataset(args.test_data_path)

    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=args.validation_batch_size,
                                              shuffle=False)
    logging.info(f"Data path: {args.test_data_path}")
    logging.info(f"Loaded dataset of {len(eval_dataset)} samples. Evaluating...")

    droplet_ids = []
    embeddings = []
    images = []

    i = 0
    for x, y in eval_loader:
        x, y = x.to(args.device), y.to(args.device)

        out_x = model(x)
        out_y = model(y)

        out_x = out_x.detach().cpu().numpy()
        out_y = out_y.detach().cpu().numpy()

        for j in range(out_x.shape[0]):
            droplet_ids += [i, i]
            embeddings += [out_x[j], out_y[j]]
            images += [wandb.Image(x[j]), wandb.Image(y[j])]
            i += 1

    df = pd.DataFrame({
        'target': droplet_ids,
        'image': images,
        'embeddings': embeddings
    })

    wandb.log({"example_small_embeddings": df})
    wandb.finish()


if __name__ == '__main__':
    main()
