import argparse
import logging
import os
import wandb
import datetime
import time


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True, help='Path to data.')
    parser.add_argument('--model', default="efficientnet-b0", help='Which model to use. Default is None.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size. Default is 32.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs. Default is 100.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate. Default is 0.001.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay. Default is 0.')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for InfoNCE loss. Default is 0.1.')
    parser.add_argument('--auroc_mode', type=str, default='roll',
                        help='Mode for calculating the negative class for AUROC')
    parser.add_argument('--wandb', action='store_true', help='Whether to use wandb for logging.')
    parser.add_argument('--checkpoint_path', default="/cluster/scratch/rdanis/checkpoints/", type=str,
                        help='Where to store model checkpoints. If not provided, the model is not stored.')
    parser.add_argument('--embed_dim', default=32, type=int, help='Dimension of the embedding layer.')
    parser.add_argument('--samples_per_epoch', default=None, type=int, help='Number of samples per epoch.')
    parser.add_argument('--validation_batch_size', default=32, type=int, help='Batch size for validation.')
    parser.add_argument('--use_dapi', action='store_true', help='Whether to use DAPI channel.')
    parser.add_argument('--train_folder', default='training', type=str, help='Name of the training folder.')

    return parser.parse_args()


def setup_logging(args: argparse.Namespace):
    
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")

    logging.basicConfig(
        filename=f"logs/training_{timestamp}.txt",
        level=logging.INFO,
        format="[%(asctime)s - %(levelname)s] %(message)s"
    )

    # set wandb environment variables so that it works on euler GPU
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    # initialize wandb
    if args.wandb:
        wandb.init(
            project="DrTrack",
            entity="dslab-23",
            config=vars(args),
            dir="logs"
        )
