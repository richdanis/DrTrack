import numpy as np
import torch
import pandas as pd
from skimage.transform import resize

from torch.utils.data import Dataset


def resize_patch_column(df):
    """
    Resizes all patches in the DataFrame to 40x40.

    Each patch in the DataFrame is resized to a 2D 40x40 patch and stored in a numpy array.

    Parameters:
    df (DataFrame): The DataFrame containing patches to be resized.

    Returns:
    numpy array: The resulting numpy array containing resized patches.
    """

    resized = np.empty((len(df), 1, 1, 40, 40))

    for i in range(len(df)):
        patch = df.iloc[i]
        patch = resize(patch, (1, 40, 40))
        patch = np.expand_dims(patch, axis=0)

        resized[i] = patch

    return resized


class DropletDataset(Dataset):

    # TODO: currently my patches are of size 40x40.
    # Should be turned into parameter.

    def __init__(self, patches_path, resize_patches=True):
        patches = np.load(patches_path, allow_pickle=True).item()
        self.patches_df = pd.DataFrame(patches)

        # For now all the embedding models are trained based on 40x40 images
        if resize_patches:
            self.patches_df['patch'] = self.patches_df['patch'].apply(lambda x: resize(x, (1, 40, 40)))


    def __len__(self):
        return len(self.patches_df)

    def __getitem__(self, idx):

        return torch.from_numpy(self.patches_df['patch'][idx]).type(torch.FloatTensor), self.patches_df['droplet_id'][idx], \
            self.patches_df['frame'][idx]
