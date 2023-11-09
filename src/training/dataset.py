import numpy as np
import torch

from torch.utils.data import Dataset


class CellDataset(Dataset):

    # TODO: currently my patches are of size 40x40.
    # Should be turned into parameter.

    def __init__(self, dataset_path, both_channels=False):
        self.dataset = np.load(dataset_path)
        # format: (N, 2, C, H, W)
        # N: number of patches
        # Second dimension: input and label (in this order)
        # C: number of channels
        # H: height
        # W: width

        # turn into torch tensor
        self.dataset = torch.from_numpy(self.dataset)

        # turn into float
        self.dataset = self.dataset.float()

        # remove one channel if necessary
        if not both_channels:
            self.dataset = self.dataset[:, :, :1, :, :]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # TODO: Augmentation?

        return self.dataset[idx, 0], self.dataset[idx, 1]
