import sys

import argparse
import logging
import os
import wandb
import datetime
import time
import torch
from info_nce import InfoNCE

sys.path.append('src/')
sys.path.append('src/training/')
from training.models.efficientnet import EfficientNet
from training.dataset import CellDataset
from training.train import evaluate
from setup import get_args, load_model


def setup_logging():
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")

    logging.basicConfig(
        filename=f"logs/evaluation_{timestamp}.txt",
        level=logging.INFO,
        format="[%(asctime)s - %(levelname)s] %(message)s"
    )


def main():
    args = get_args()
    setup_logging()

    model = load_model(args)
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(args.device))
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Model path: {args.checkpoint_path}")
    logging.info(f"Model loaded.")

    eval_dataset = CellDataset(args.test_data_path)

    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=723,
                                              shuffle=True)
    logging.info(f"Data path: {args.test_data_path}")
    logging.info(f"Loaded dataset of {len(eval_dataset)} samples. Evaluating...")
    criterion = InfoNCE()
    eval_log_dict = evaluate(model, eval_loader, criterion, epoch=-1, args=args)
    logging.info(eval_log_dict)


if __name__ == '__main__':
    main()
