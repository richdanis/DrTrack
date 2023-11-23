import logging
from tqdm import tqdm
import numpy as np
import wandb
import torch
from info_nce import InfoNCE

class Trainer():

    def __init__(self, 
                 model, 
                 optimizer, 
                 loss_fn, 
                 train_loader, 
                 val_loader,  
                 device,
                 val_labels,
                 val_neg_ind,
                 save_path, 
                 args):
        
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.val_labels = val_labels
        self.val_neg_ind = val_neg_ind
        self.save_path = save_path
        self.args = args

        self.val_losses = []
        self.val_acc_top1 = []
        self.val_acc_top5 = []

        self.best_top5_acc = 0

    def train(self):
        for epoch in range(self.args.epochs):
            logging.info(f"Epoch {epoch+1} of {self.args.epochs}")
            self.train_epoch(epoch)
            self.val_epoch(epoch)
            self.log()

            self.save_model(epoch)

        if self.args.wandb:
            wandb.finish()

    def log(self):

        logging.info(f"Val Loss: {self.val_losses[-1]:.4f}")
        logging.info(f"Val Acc Top 1: {self.val_acc_top1[-1]:.4f}")
        logging.info(f"Val Acc Top 5: {self.val_acc_top5[-1]:.4f}")
        if self.args.wandb:
            wandb.log({
                'val_loss': self.val_losses[-1],
                'val_acc_top1': self.val_acc_top1[-1],
                'val_acc_top5': self.val_acc_top5[-1]
            })

    def train_epoch(self, epoch):

        # count num samples for epoch
        count = 0

        self.model.train()

        criterion = InfoNCE()
        for x, y in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs}"):
            x, y = x.to(self.device), y.to(self.device)

            # TODO: keep track of embeddings to avoid recomputing them
            embeddings_x = self.model(x)
            embeddings_y = self.model(y)

            # output = self.loss_fn(embeddings_x, embeddings_y, embeddings_z)
            output = criterion(embeddings_x, embeddings_y)

            self.optimizer.zero_grad()
            output.backward()
            self.optimizer.step()

            if self.args.samples_per_epoch is not None:
                count += self.args.batch_size
                if count > self.args.samples_per_epoch:
                    break

    def val_epoch(self, epoch):

        self.model.eval()

        embeddings_x = torch.empty((0, self.args.embed_dim))

        for x in tqdm(self.val_loader, desc=f"Val epoch {epoch + 1}/{self.args.epochs}"):
            
            with torch.no_grad():
                
                x = x.to(self.device)

                out_x = self.model(x).cpu()

            embeddings_x = torch.cat((embeddings_x, out_x), dim=0)

        # TODO: factor this out into a function
        top1_accuracy = []
        top5_accuracy = []
        losses = []

        # TODO: could batch this to make it faster
        for i in range(self.val_labels.shape[0]):
            positive = embeddings_x[self.val_labels[i]]
            negatives = embeddings_x[self.val_neg_ind[i]]

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
                losses.append(self.loss_fn(embeddings_x[i].unsqueeze(0), 
                                           positive.unsqueeze(0), 
                                           negatives.unsqueeze(0)).item())
        
        self.val_losses.append(np.mean(losses))
        self.val_acc_top1.append(np.sum(top1_accuracy) / len(top1_accuracy))
        self.val_acc_top5.append(np.sum(top5_accuracy) / len(top5_accuracy))

    def save_model(self, epoch):

        if self.save_path is not None and self.best_top5_acc < self.val_acc_top5[-1]:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.val_losses[-1],
                }, self.args.model_path)
                logging.info(
                    f'Validation top5 accuracy improved from {self.best_top5_acc} to {self.val_acc_top5[-1]}. Saved the model.')
                self.best_top5_acc = self.val_acc_top5[-1]
