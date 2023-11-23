import datetime
import os
import time

import wandb
import torch
import argparse
import logging
import numpy as np
from info_nce import InfoNCE
from tqdm import tqdm
from dataset import LocalDataset
from models.efficientnet import EfficientNet

from metrics.auroc import calculate_auroc
from metrics.accuracy import calculate_accuracy
from utils.setup import setup_logging, get_args
from trainer import Trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # get arguments
    args = get_args()

    # initialize logging and wandb
    setup_logging(args)

    # prepare for saving the model
    if args.checkpoint_path is not None:
        timestamp = datetime.datetime.fromtimestamp(
            time.time()).strftime("%Y-%m-%d_%H-%M-%S")
        model_path = os.path.join(
            args.checkpoint_path, f'{timestamp}_embeddings_model.pt')
    else:
        model_path = None

    # load datasets
    train_dataset = LocalDataset(os.path.join(args.data_path, 'training'),
                                 config='train')
    val_dataset = LocalDataset(os.path.join(args.data_path, 'validation'),
                               config='val')
    
    val_labels, val_neg_ind = val_dataset.labels, val_dataset.negatives
    logging.info(f"Datasets loaded.")

    # data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.validation_batch_size,
                                             shuffle=False)

    # load model
    model = load_model(args)
    model = model.to(DEVICE)
    logging.info(f"Model loaded.")

    # log number of parameters
    logging.info(
        f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # loss criterion
    # https://github.com/RElbers/info-nce-pytorch
    # this is just a first try, can maybe use different contrastive loss
    criterion = InfoNCE(negative_mode='paired')

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      loss_fn=criterion,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      device=DEVICE,
                      val_labels=val_labels,
                      val_neg_ind=val_neg_ind,
                      save_path=model_path,
                      args=args)

    trainer.train()

    if args.wandb:
        wandb.finish()


def load_model(args):
    # TODO: needs to be changed if we want to be able to load different models.
    # Currently using EfficientNet, version b1, lightweight: 6.5 million parameters.
    # Output embedding has 1280 dimensions.
    return EfficientNet(args)


if __name__ == "__main__":
    main()
