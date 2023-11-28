import logging
from tqdm import tqdm
import numpy as np
import wandb
import torch
from info_nce import InfoNCE
import time
import datetime
import os


class Trainer():

    def __init__(self,
                 model,
                 optimizer,
                 loss_fn,
                 train_loader,
                 train_eval_loader,
                 val_loader,
                 device,
                 train_indices,
                 val_indices,
                 args):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.train_eval_loader = train_eval_loader
        self.val_loader = val_loader
        self.device = device
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.args = args

        self.best_top5_acc = 0

        self.val_loss = InfoNCE(negative_mode='paired')

        # prepare for saving the model
        self.model_path = None
        if args.checkpoint_path is not None:
            timestamp = datetime.datetime.fromtimestamp(
                time.time()).strftime("%Y-%m-%d_%H-%M-%S")
            self.model_path = os.path.join(
                args.checkpoint_path, f'{timestamp}_dim_{args.embed_dim}.pth')

    def train(self):
        for epoch in range(self.args.epochs):
            logging.info(f"Epoch {epoch+1} of {self.args.epochs}")
            self.train_epoch(epoch)
            train_eval_dict = self.val_epoch(
                loader=self.train_eval_loader, epoch=epoch, **self.train_indices)
            val_dict = self.val_epoch(
                loader=self.val_loader, epoch=epoch, **self.val_indices)
            self.log(train_eval_dict, val_dict)

            self.save_model(epoch, val_dict)

        if self.args.wandb:
            wandb.finish()

    def log(self, train_eval_dict, val_dict):

        # log dicts with keys capitalized
        logging.info(f"Train: {train_eval_dict}")
        logging.info(f"Val: {val_dict}")

        # append train/val to keys
        train_eval_dict = {f"train_{k}": v for k, v in train_eval_dict.items()}
        val_dict = {f"val_{k}": v for k, v in val_dict.items()}
        if self.args.wandb:
            wandb.log({
                **train_eval_dict,
                **val_dict
            })

    def train_epoch(self, epoch):

        # count num samples for epoch
        count = 0

        self.model.train()

        for x, y in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs}"):
            x, y = x.to(self.device), y.to(self.device)

            # TODO: keep track of embeddings to avoid recomputing them
            embeddings_x = self.model(x)
            embeddings_y = self.model(y)

            # output = self.loss_fn(embeddings_x, embeddings_y, embeddings_z)
            output = self.loss_fn(embeddings_x, embeddings_y)

            self.optimizer.zero_grad()
            output.backward()
            self.optimizer.step()

            if self.args.samples_per_epoch is not None:
                count += self.args.batch_size
                if count > self.args.samples_per_epoch:
                    break

    def val_epoch(self, loader, epoch, labels, neg_ind):

        self.model.eval()

        embeddings_x = torch.empty((0, self.args.embed_dim))

        for x in tqdm(loader, desc=f"Val epoch {epoch + 1}/{self.args.epochs}"):

            with torch.no_grad():

                x = x.to(self.device)

                out_x = self.model(x).cpu()

            embeddings_x = torch.cat((embeddings_x, out_x), dim=0)

        # TODO: factor this out into a function
        top1_accuracy = []
        top5_accuracy = []
        losses = []

        # TODO: could batch this to make it faster
        for i in range(labels.shape[0]):
            positive = embeddings_x[labels[i]]
            negatives = embeddings_x[neg_ind[i]]

            # compute distances
            distances = np.linalg.norm(embeddings_x[i] - negatives, axis=1)
            distances_pos = np.linalg.norm(embeddings_x[i] - positive)
            distances = np.append(distances, distances_pos)

            # label is appended to the end
            target_ind = distances.shape[0] - 1

            # sort distances
            distances_sort = np.argsort(distances)

            # check if the matched droplet is in the topk
            top1_accuracy.append(int(target_ind in distances_sort[:1]))
            top5_accuracy.append(int(target_ind in distances_sort[:5]))

            # compute the loss
            with torch.no_grad():
                losses.append(self.val_loss(embeddings_x[i].unsqueeze(0),
                                            positive.unsqueeze(0),
                                            negatives.unsqueeze(0)).item())

        return {
            "loss": np.mean(losses),
            "acc_top1": np.sum(top1_accuracy) / len(top1_accuracy),
            "acc_top5": np.sum(top5_accuracy) / len(top5_accuracy)
        }

    def save_model(self, epoch, val_dict):

        if self.model_path is not None and self.best_top5_acc < val_dict['acc_top5']:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': val_dict['loss'],
            }, self.model_path)
            logging.info(
                f'Validation top5 accuracy improved from {self.best_top5_acc} to {val_dict["acc_top5"]}. Saved the model.')
            self.best_top5_acc = val_dict['acc_top5']
