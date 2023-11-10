import os

import wandb
import torch
import argparse
import torchvision
import logging
from info_nce import InfoNCE
from tqdm import tqdm
from dataset import CellDataset
from models.efficientnet import EfficientNet
import numpy as np

from metrics.auroc import calculate_auroc
from metrics.accuracy import calculate_accuracy


def main():
    parser = argparse.ArgumentParser()
    args = get_args(parser)

    logging.basicConfig(filename='logs/training.log', level=logging.DEBUG)

    # load datasets
    train_dataset = CellDataset(os.path.join(args.data_path, 'train.npy'))
    val_dataset = CellDataset(os.path.join(args.data_path, 'validation.npy'))
    logging.info(f"Dataset loaded.")

    # data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True)

    # initialize wandb
    if args.wandb:
        wandb.init(
            project="dsl",
            config=vars(args)
        )

    # load model
    model = load_model()
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

    for epoch in range(args.epochs):
        train_log_dict = train_one_epoch(model, train_loader, optimizer, criterion, epoch, args)
        val_log_dict = evaluate(model, val_loader, criterion, epoch, args)
        if args.wandb:
            wandb.log({**train_log_dict, **val_log_dict})

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
    # TODO: Logging, keeping track of loss, metrics, etc.

    # progress bar
    pbar = tqdm(train_dl, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")
    pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}")

    # log epoch
    logging.info(f"Epoch {epoch}")

    # keep track of loss for each iteration
    info_loss = []

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

    info_loss = []
    info_topk_accuracy = []
    info_auroc = []
    for x, y in val_dl:
        x, y = x.to(args.device), y.to(args.device)

        embeddings_x = model(x)
        embeddings_y = model(y)

        output = criterion(embeddings_x, embeddings_y)
        info_loss.append(output.item())

        embeddings_x = embeddings_x.detach().numpy()
        embeddings_y = embeddings_y.detach().numpy()

        info_topk_accuracy.append(calculate_accuracy(embeddings_x, embeddings_y, args.topk_accuracy))

        info_auroc.append(calculate_auroc(embeddings_x, embeddings_y, args.auroc_mode))

    mean_loss = sum(info_loss) / len(info_loss)
    mean_auroc = sum(info_auroc) / len(info_auroc)
    logging.info(f"Val loss: {mean_loss}")
    logging.info(f"Val auroc: {mean_auroc}")

    log_dict = {
        'val_mean_epoch_loss': mean_loss,
        'val_mean_auroc': mean_auroc,
    }

    mean_topk_accuracy = np.mean(info_topk_accuracy, axis=0)
    for k, acc in zip(args.topk_accuracy, mean_topk_accuracy):
        log_dict[f'val_mean_top{k}_accuracy'] = acc
        logging.info(f"Val mean top {k} accuracy: {acc}")

    return log_dict


def get_args(parser):
    # TODO: add more arguments

    parser.add_argument('--data_path', type=str, required=True, help='Path to data.')
    parser.add_argument('--model', default=None, help='Which model to use. Default is None.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size. Default is 32.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs. Default is 100.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate. Default is 0.001.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use. Default is cuda.')
    parser.add_argument('--topk_accuracy', type=int, nargs='+', help='Whether to log validation topk accuracy.')
    parser.add_argument('--auroc_mode', type=str, default='roll',
                        help='Mode for calculating the negative class for AUROC')
    parser.add_argument('--wandb', type=bool, default=False, help='Whether to log to wandb.')

    return parser.parse_args()


def load_model():
    # TODO: needs to be changed if we want to be able to load different models.
    # Currently using EfficientNet, version b1, lightweight: 6.5 million parameters.
    # Output embedding has 1280 dimensions.
    return EfficientNet()


if __name__ == "__main__":
    main()
