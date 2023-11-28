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
from tqdm import tqdm

from dataset import VisualizationDataset
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

    eval_dataset = VisualizationDataset(args.test_data_path, args.test_metadata_path)

    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=args.validation_batch_size,
                                              shuffle=False)
    logging.info(f"Data path: {args.test_data_path}")
    logging.info(f"Loaded dataset of {len(eval_dataset)} samples. Evaluating...")

    num_cells = []
    droplet_ids = []
    embeddings = []
    images = []
    segments = []
    center_rows = []
    center_cols = []

    for x, y in tqdm(eval_loader):
        patch_batch, metadata_batch = x.to(args.device), y
        patch_batch = patch_batch.type(torch.FloatTensor)
        with torch.no_grad():

            embedding_batch = model(patch_batch)

            embedding_batch = embedding_batch.detach().cpu().numpy()

        for j in range(embedding_batch.shape[0]):
            droplet_ids.append(metadata_batch['droplet_id'][j])
            num_cells.append(metadata_batch['num_cells'][j])
            segments.append(metadata_batch['segment'][j])
            center_rows.append(metadata_batch['center_row'][j])
            center_cols.append(metadata_batch['center_col'][j])
            embeddings.append(embedding_batch[j])
            images.append(wandb.Image(patch_batch[j]))

    df = pd.DataFrame({
        'droplet_id': droplet_ids,
        'nr_cells': num_cells,
        'image': images,
        'embeddings': embeddings,
        'segments': segments,
        'center_rows': center_rows,
        'center_cols': center_cols
    })

    wandb.log({"example_small_embeddings": df})
    wandb.finish()


if __name__ == '__main__':
    main()
