import os

import wandb
import torch
import logging
from info_nce import InfoNCE
from dataset import LocalDataset
from models.efficientnet import EfficientNet

from utils.setup import setup_logging, get_args
from trainer import Trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # get arguments
    args = get_args()

    # initialize logging and wandb
    setup_logging(args)

    # load datasets
    train_dataset = LocalDataset(os.path.join(args.data_path, args.train_folder),
                                 config='train', use_dapi=args.use_dapi)
    train_eval_dataset = LocalDataset(os.path.join(args.data_path, args.train_folder),
                                      config='val', use_dapi=args.use_dapi)
    val_dataset = LocalDataset(os.path.join(args.data_path, 'validation'),
                               config='val', use_dapi=args.use_dapi)
    
    logging.info(f"Datasets loaded.")

    # data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    train_eval_loader = torch.utils.data.DataLoader(train_eval_dataset,
                                                    batch_size=args.validation_batch_size,
                                                    shuffle=False)
    train_indices = {
        'labels': train_eval_dataset.labels,
        'neg_ind': train_eval_dataset.negatives
    }
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.validation_batch_size,
                                             shuffle=False)
    val_indices = {
        'labels': val_dataset.labels,
        'neg_ind': val_dataset.negatives
    }

    # load model
    model = load_model(args)
    model = model.to(DEVICE)
    logging.info(f"Model loaded.")

    # log number of parameters
    logging.info(
        f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # loss criterion
    # https://github.com/RElbers/info-nce-pytorch
    # this is just a first try, can maybe use different contrastive loss
    criterion = InfoNCE(temperature=args.temperature)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      loss_fn=criterion,
                      train_loader=train_loader,
                      train_eval_loader=train_eval_loader,
                      val_loader=val_loader,
                      device=DEVICE,
                      train_indices=train_indices,
                      val_indices=val_indices,
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
