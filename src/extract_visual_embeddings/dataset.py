import numpy as np
import torch
import pandas as pd
from skimage.transform import resize

from torch.utils.data import Dataset


class DropletDataset(Dataset):

    # TODO: currently my patches are of size 40x40.
    # Should be turned into parameter.

    def __init__(self, patches_path, droplet_path, resize_patches=True):
        patches = np.load(patches_path, allow_pickle=True).item()
        self.patches_df = pd.DataFrame(patches)
        droplets_df = pd.read_csv(droplet_path)
        # For simulated data we can have some difference between the number of droplets in patches_df and droplets_df.
        # For real experiments it should be exactly the same, because one is constructed based on the other.
        self.patches_df = pd.merge(self.patches_df, droplets_df, on=['droplet_id', 'frame'], how='inner')

        # For now all the embedding models are trained based on 40x40 images
        if resize_patches:
            for j in range(len(self.patches_df)):
                patch = self.patches_df.at[j, 'patch']
                patch = resize(patch, (2, 40, 40))
                self.patches_df.at[j, 'patch'] = patch

    def __len__(self):
        return len(self.patches_df)

    def __getitem__(self, idx):

        return torch.from_numpy(self.patches_df['patch'][idx]).type(torch.FloatTensor), self.patches_df['droplet_id'][
            idx], \
            self.patches_df['frame'][idx], self.patches_df['nr_cells'][idx]
