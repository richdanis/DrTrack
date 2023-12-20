import numpy as np
import torch
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from skimage.transform import resize


class PairedDataset(Dataset):

    def __init__(self, dataset_path, size, frame_num=0):

        all_patches = pd.DataFrame(np.load(dataset_path, allow_pickle=True).item())
        self.patches = all_patches[((all_patches['frame'] == frame_num) | (all_patches['frame'] == frame_num + 1)) & (all_patches['droplet_id'] < size)]
        self.frame_num = frame_num
        print(len(self.patches) // 2)

    def __len__(self):
        return len(self.patches) // 2

    def __getitem__(self, idx):
        # TODO: Debug and extract new patches
        patch = self.patches[self.patches['droplet_id'] == idx]
        patch1 = resize(patch[patch['frame'] == self.frame_num]['patch'].item(), (2, 40, 40))
        patch2 = resize(patch[patch['frame'] == self.frame_num + 1]['patch'].item(), (2, 40, 40))
        return torch.from_numpy(patch1).type(torch.FloatTensor), torch.from_numpy(patch2).type(torch.FloatTensor)


class CellDataset(Dataset):

    # TODO: currently my patches are of size 40x40.
    # Should be turned into parameter.

    def __init__(self, dataset_path, use_dapi=False):
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
        if not use_dapi:
            self.dataset = self.dataset[:, :, :1, :, :]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # TODO: Augmentation?

        return self.dataset[idx, 0], self.dataset[idx, 1]


class LocalDataset(Dataset):

    def __init__(self, dataset_path, config, use_dapi=False):

        self.patches = np.load(dataset_path + '/patches.npy')
        # format: (N, C, H, W)
        # N: number of patches
        # Second dimension: input and label (in this order)
        # C: number of channels
        # H: height
        # W: width

        self.labels = np.load(dataset_path + '/labels.npy')
        # format: (M,), where M < N
        # since we have less labels then patches, because patches
        # of last image have no label (no next image)

        # self.negatives = np.load(dataset_path + '/negatives.npy')
        # # format: (M, 128)

        # train or validation
        self.config = config

        # remove one channel (done by default)
        if not use_dapi:
            self.patches = self.patches[:, :1, :, :]

    def __len__(self):

        if self.config == 'val':
            return len(self.patches)
        return len(self.labels)

    def __getitem__(self, idx):
        
        if self.config == 'val':
            return self.patches[idx]
        
        x = self.patches[idx]
        y = self.patches[self.labels[idx]]

        return x, y
