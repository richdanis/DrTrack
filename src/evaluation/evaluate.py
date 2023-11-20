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


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_data_path', type=str, default=None, help='Full path to .npy file with data.')
    parser.add_argument('--model', default=None, help='Which model to use. Default is None.')
    parser.add_argument('--topk_accuracy', type=int, nargs='+', help='Whether to log validation topk accuracy.')
    parser.add_argument('--auroc_mode', type=str, default='roll',
                        help='Mode for calculating the negative class for AUROC')
    parser.add_argument('--checkpoint_path', required=True, type=str,
                        help='Where model checkpoint is stored.')
    parser.add_argument('--validation_batch_size', type=int, default=32, help='Batch size. Default is 32.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use. Default is cpu.')
    parser.add_argument('--embed_dim', default=None, type=int, help='Dimension of the embedding layer.')

    return parser.parse_args()


def load_model(args: argparse.Namespace):
    # TODO: needs to be changed if we want to be able to load different models.
    # Currently using EfficientNet, version b1, lightweight: 6.5 million parameters.
    # Output embedding has 1280 dimensions.
    return EfficientNet(args)


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
    logging.info(f"Model loaded.")

    eval_dataset = CellDataset(args.test_data_path)

    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=args.validation_batch_size,
                                              shuffle=True)
    logging.info(f"Loaded dataset of {len(eval_dataset)} samples. Evaluating...")
    criterion = InfoNCE()
    eval_log_dict = evaluate(model, eval_loader, criterion, epoch=-1, args=args)
    logging.info(eval_log_dict)


if __name__ == '__main__':
    main()
