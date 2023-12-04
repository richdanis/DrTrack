import sys

import logging
import wandb
import datetime
import time
import torch
from info_nce import InfoNCE

sys.path.append('src/')
sys.path.append('src/training/')
from training.models.efficientnet import EfficientNet
from training.dataset import LocalDataset
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

    # need to be adapted to local datasets
    args = get_args()
    setup_logging()

    model = EfficientNet(args)
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(args.device))
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Model path: {args.checkpoint_path}")
    logging.info(f"Model loaded.")

    eval_dataset = LocalDataset(args.test_data_path,
                                config='val', 
                                use_dapi=args.use_dapi)

    eval_indices = {
        'labels': eval_dataset.labels,
        'neg_ind': eval_dataset.negatives
    }

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
    
    eval_log_dict = evaluator.val_epoch(loader=eval_loader, epoch=0, **eval_indices)
    logging.info(eval_log_dict)


if __name__ == '__main__':
    main()
