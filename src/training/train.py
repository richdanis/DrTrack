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
from dataset import CellDataset
from models.efficientnet import EfficientNet

from metrics.auroc import calculate_auroc
from metrics.accuracy import calculate_accuracy
from utils.setup import setup_logging, get_args


def main():
    # get arguments
    args = get_args()

    # initialize logging and wandb
    setup_logging(args)

    # prepare for saving the model
    if args.checkpoint_path is not None:
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
        model_path = os.path.join(args.checkpoint_path, f'{timestamp}_embeddings_model.pt')
    else:
        model_path = None

    # load datasets
    train_dataset = CellDataset(os.path.join(args.data_path, 'train.npy'))
    val_dataset = CellDataset(os.path.join(args.data_path, 'validation.npy'))
    logging.info(f"Dataset loaded.")

    # data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.validation_batch_size,
                                             shuffle=True)

    # load model
    model = load_model(args)
    model = model.to(args.device)
    logging.info(f"Model loaded.")

    # log number of parameters
    logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # loss criterion
    # https://github.com/RElbers/info-nce-pytorch
    # this is just a first try, can maybe use different contrastive loss
    criterion = InfoNCE()

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_log_dict = train_one_epoch(model, train_loader, optimizer, criterion, epoch, args)
        val_log_dict = evaluate(model, val_loader, criterion, epoch, args)
        if args.wandb:
            wandb.log({**train_log_dict, **val_log_dict})

        if model_path is not None and best_val_loss > val_log_dict['val_mean_epoch_loss']:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_log_dict['train_mean_epoch_loss'],
            }, model_path)
            logging.info(
                f'Validation loss improved from {best_val_loss} to {val_log_dict["val_mean_epoch_loss"]}. Saved the model.')
            best_val_loss = val_log_dict['val_mean_epoch_loss']

    if args.wandb:
        wandb.finish()


def train_one_epoch(
        model: torch.nn.Module,
        train_dl: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        epoch: int,
        args: argparse.Namespace
):

    # progress bar
    pbar = tqdm(train_dl, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")
    pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}")

    # log epoch
    logging.info(f"Epoch {epoch}")

    # keep track of loss for each iteration
    info_loss = []

    # count num samples for epoch
    count = 0

    model.train()
    for x, y in train_dl:
        x, y = x.to(args.device), y.to(args.device)

        embeddings_x = model(x)
        embeddings_y = model(y)

        output = criterion(embeddings_x, embeddings_y)

        optimizer.zero_grad()
        output.backward()
        optimizer.step()

        # update progress bar with current loss
        pbar.set_postfix(
            loss=f"{output.item():.4f}",
        )
        pbar.update()

        info_loss.append(output.item())

        if args.samples_per_epoch is not None:
            count += args.batch_size
            if count > args.samples_per_epoch:
                break

    # log average loss for epoch
    mean_loss = sum(info_loss) / len(info_loss)
    logging.info(f"Loss: {mean_loss}")

    # TODO: Consider adding other metrics also for training
    log_dict = {
        'train_mean_epoch_loss': mean_loss
    }

    return log_dict


def evaluate(
        model: torch.nn.Module,
        val_dl: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        epoch: int,
        args: argparse.Namespace
):
    # log epoch info
    logging.info(f"Evaluating after epoch {epoch}")

    # prepare topk accuracy params
    topk = set() if args.topk_accuracy is None else set(args.topk_accuracy)
    topk.add(1)  # we want to always list top1 accuracy
    topk = list(topk)

    info_loss = []

    # put model to eval mode
    model.eval()

    # arrays for x and y embeddings
    embeddings_x = np.empty((0, args.embed_dim))
    embeddings_y = np.empty((0, args.embed_dim))

    for x, y in val_dl:
        x, y = x.to(args.device), y.to(args.device)

        out_x = model(x)
        out_y = model(y)

        output = criterion(out_x, out_y)
        info_loss.append(output.item())

        out_x = out_x.detach().cpu().numpy()
        out_y = out_y.detach().cpu().numpy()

        embeddings_x = np.concatenate((embeddings_x, out_x), axis=0)
        embeddings_y = np.concatenate((embeddings_y, out_y), axis=0)

    info_topk_accuracy = calculate_accuracy(embeddings_x, embeddings_y, topk)
    info_auroc = calculate_auroc(embeddings_x, embeddings_y, args.auroc_mode)
    
    mean_loss = sum(info_loss) / len(info_loss)
    logging.info(f"Val loss: {mean_loss}")
    logging.info(f"Val auroc: {info_auroc}")

    log_dict = {
        'val_mean_epoch_loss': mean_loss,
        'val_auroc': info_auroc,
    }

    for k, acc in zip(topk, info_topk_accuracy):
        log_dict[f'val_top{k}_accuracy'] = acc
        logging.info(f"Val top {k} accuracy: {acc}")

    return log_dict


def load_model(args):
    # TODO: needs to be changed if we want to be able to load different models.
    # Currently using EfficientNet, version b1, lightweight: 6.5 million parameters.
    # Output embedding has 1280 dimensions.
    return EfficientNet(args)


if __name__ == "__main__":
    main()
