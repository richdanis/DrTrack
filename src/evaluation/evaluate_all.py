import sys

import logging
import wandb
import datetime
import time
import torch
from info_nce import InfoNCE
import numpy as np

sys.path.append('src/')
sys.path.append('src/training/')
from training.models.efficientnet import EfficientNet
from training.dataset import PairedDataset
from training.trainer import Trainer
from setup import get_args


def setup_logging():
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")

    logging.basicConfig(
        filename=f"logs/evaluation_{timestamp}.txt",
        level=logging.INFO,
        format="[%(asctime)s - %(levelname)s] %(message)s"
    )


def main():
    np.random.seed(0)
    # need to be adapted to local datasets
    args = get_args()
    setup_logging()

    model = EfficientNet(args)
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(args.device))
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Model path: {args.checkpoint_path}")
    logging.info(f"Model loaded.")
    results = {
        'acc_top1_l2': [],
        'acc_top5_l2': [],
        'auroc_l2': [],
        'acc_top1_cosine': [],
        'acc_top5_cosine': [],
        'auroc_cosine': [],
    }

    for frame_num in range(8):
        eval_dataset = PairedDataset(dataset_path=args.test_data_path, size=args.sample_size, frame_num=frame_num)

        eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                                  batch_size=128,
                                                  shuffle=False)

        logging.info(f"Data path: {args.test_data_path}")
        logging.info(f"Loaded dataset of {len(eval_dataset)} samples. Evaluating...")
        criterion = InfoNCE()

        args.epochs = 1

        evaluator = Trainer(model=model,
                            optimizer=None,
                            loss_fn=criterion,
                            train_loader=None,
                            train_eval_loader=None,
                            val_loader=None,
                            device=args.device,
                            train_indices=None,
                            val_indices=None,
                            args=args)
        evaluator.full_val_epoch(loader=eval_loader, results=results)
    for k in results.keys():
        print(k, np.mean(results[k]), np.std(results[k]) / np.sqrt(len(results[k])))


if __name__ == '__main__':
    main()
