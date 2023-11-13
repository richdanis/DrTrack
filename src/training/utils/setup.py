import argparse
import logging
import os
import wandb
import datetime
import time


def get_args():
    # TODO: add more arguments

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True, help='Path to data.')
    parser.add_argument('--model', default=None, help='Which model to use. Default is None.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size. Default is 32.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs. Default is 100.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate. Default is 0.001.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use. Default is cuda.')
    parser.add_argument('--topk_accuracy', type=int, nargs='+', help='Whether to log validation topk accuracy.')
    parser.add_argument('--auroc_mode', type=str, default='roll',
                        help='Mode for calculating the negative class for AUROC')
    parser.add_argument('--wandb', action='store_true', help='Whether to use wandb for logging.')
    parser.add_argument('--checkpoint_path', default=None, type=str,
                        help='Where to store model checkpoints. If not provided, the model is not stored.')
    parser.add_argument('--embed_dim', default=None, type=int, help='Dimension of the embedding layer.')
    parser.add_argument('--samples_per_epoch', default=None, type=int, help='Number of samples per epoch.')
    parser.add_argument('--validation_batch_size', default=32, type=int, help='Batch size for validation.')

    return parser.parse_args()


def setup_logging(args: argparse.Namespace):
    # TODO: create different directory for each run
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")

    logging.basicConfig(
        filename=f"logs/training_{timestamp}.txt",
        level=logging.INFO,
        format="[%(asctime)s - %(levelname)s] %(message)s"
    )

    # set wandb environment variables so that it workes on euler GPU
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    # initialize wandb
    if args.wandb:
        wandb.init(
            project="DrTrack",
            entity="dslab-23",
            config=vars(args),
            dir="logs"
        )
