# Types
from pathlib import Path
from omegaconf import DictConfig
from torch.utils.data import Dataset

# Data handling
import numpy as np
import torch
import pandas as pd
from skimage.transform import resize


class DropletDataset(Dataset):

    # TODO: currently my patches are of size 40x40.
    # Should be turned into parameter.

    def __init__(self, 
                 cfg: DictConfig,
                 patches_path: Path, 
                 droplet_path: Path, 
                 resize_patches: bool = True):
        """
        Parameters
        ----------
        patches_path : Path
            Path to the patches file.
        droplet_path : Path
            Path to the droplet file.
        resize_patches : bool, optional
            Whether to resize the patches to 40x40. Defaults to True.
        
        Attributes
        ----------
        patches_df : pandas.DataFrame
            DataFrame containing the patches.
        
        Methods
        -------
        __len__()
            Return the length of the dataset.
        __getitem__(idx)
            Return the item at the given index.
        """
        patches = np.load(patches_path, allow_pickle=True).item()
        self.patches_df = pd.DataFrame(patches)

        # To reduce the size of the dataset for evaluation purposes
        #self.patches_df = self.patches_df[self.patches_df["droplet_id"] <= 22000]

        droplets_df = pd.read_csv(droplet_path)
        # For simulated data we can have some difference between the number of droplets in patches_df and droplets_df.
        # For real experiments it should be exactly the same, because one is constructed based on the other.
        self.patches_df = pd.merge(self.patches_df, droplets_df, on=['droplet_id', 'frame'], how='inner')

        # Make sure that patch sizes are the same as for training.
        if resize_patches:
            patch_size_x = cfg.extract_visual_embeddings.patch_size_x
            patch_size_y = cfg.extract_visual_embeddings.patch_size_y
            for j in range(len(self.patches_df)):
                patch = self.patches_df.at[j, 'patch']
                patch = resize(patch, (2, patch_size_x, patch_size_y))
                self.patches_df.at[j, 'patch'] = patch

    def __len__(self):
        return len(self.patches_df)

    def __getitem__(self, idx):

        return torch.from_numpy(self.patches_df['patch'][idx]).type(torch.FloatTensor), self.patches_df['droplet_id'][
            idx], \
            self.patches_df['frame'][idx], self.patches_df['nr_cells'][idx]
