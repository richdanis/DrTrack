import sys

import argparse
import logging
import os
import wandb
import datetime
import time
import torch

sys.path.append('src')
from training.models.efficientnet import EfficientNet


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_data_path', type=str, default=None, help='Path to data.')
    parser.add_argument('--model', default=None, help='Which model to use. Default is None.')
    parser.add_argument('--auroc_mode', type=str, default='roll',
                        help='Mode for calculating the negative class for AUROC')
    parser.add_argument('--checkpoint_path', required=True, type=str,
                        help='Where to store model checkpoints. If not provided, the model is not stored.')

    return parser.parse_args()


def load_model():
    # TODO: needs to be changed if we want to be able to load different models.
    # Currently using EfficientNet, version b1, lightweight: 6.5 million parameters.
    # Output embedding has 1280 dimensions.
    return EfficientNet()


def main():
    args = get_args()

    model = load_model()
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(model)


if __name__ == '__main__':
    main()
