import torch
import argparse
import torchvision
import logging
from info_nce import InfoNCE
from tqdm import tqdm
from dataset import CellDataset
from models.efficientnet import EfficientNet


def main():

    # TODO: wandb integration

    parser = argparse.ArgumentParser()
    args = get_args(parser)

    logging.basicConfig(filename='logs/training.log', encoding='utf-8', level=logging.DEBUG)

    # load dataset
    train_dataset = CellDataset(args.data_path + 'train.npy')
    logging.info(f"Dataset loaded.")

    # data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, \
                                              batch_size=args.batch_size, \
                                              shuffle=True)
    
    # load model
    model = load_model()
    model = model.to(args.device)
    logging.info(f"Model loaded.")

    # log number of parameters
    logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # loss criterion
    # https://github.com/RElbers/info-nce-pytorch
    # this is just a first try, can maybe use different contrastive loss
    criterion = InfoNCE()

    for epoch in range(args.epochs):

        train_one_epoch(model, train_loader, optimizer, criterion, epoch, args)


def train_one_epoch(
        model: torch.nn.Module,
        train_dl: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        epoch: int,
        args: argparse.Namespace
    ):
        # TODO: Logging, keeping track of loss, metrics, etc.

        # progress bar
        pbar = tqdm(train_dl, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")
        pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}")

        # log epoch
        logging.info(f"Epoch {epoch}")

        # keep track of loss for each iteration
        info_loss = []

        model.train()

        for x,y in train_dl:

            x, y = x.to(args.device), y.to(args.device)

            embeddings_x = model(x)
            embeddings_y = model(y)

            output = criterion(embeddings_x, embeddings_y)

            optimizer.zero_grad()
            output.backward()
            optimizer.step()

            # update progress bar with current loss
            pbar.set_postfix(
                loss=f"{output.item():.4f}",
            )
            pbar.update()

            info_loss.append(output.item())

        # log average loss for epoch
        logging.info(f"Loss: {sum(info_loss) / len(info_loss)}")

def evaluate(
        model: torch.nn.Module,
        val_dl: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        epoch: int,
        args: argparse.Namespace
    ):
     
     # TODO: how to evaluate? Maybe just use accuracy.
     
     pass

def get_args(parser):

    # TODO: add more arguments

    parser.add_argument('--data_path', type=str, required=True, help='Path to data.')
    parser.add_argument('--model', default=None, help='Which model to use. Default is None.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size. Default is 32.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs. Default is 100.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate. Default is 0.001.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use. Default is cuda.')

    return parser.parse_args()

def load_model():

    # TODO: needs to be changed if we want to be able to load different models.
    # Currently using EfficientNet, version b1, lightweight: 6.5 million parameters.
    # Output embedding has 1280 dimensions.
    return EfficientNet()
    

if __name__ == "__main__":
    main()
