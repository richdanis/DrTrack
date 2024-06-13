# Types and os
import sys
from pathlib import Path
import os
from typing import Dict
from omegaconf import DictConfig

# Arrays and data frames
import numpy as np
import pandas as pd

# Computer Vision
import cv2 as cv

# Add src to path
sys.path.append('./src/')

# IMPORTANT! Default params are compatible with the current embeddings. Changing them will harm the performance.
def get_patch(image, 
              center_x: float, 
              center_y: float, 
              radius: int, 
              buffer: int = -2, 
              suppress_rest: bool = True, 
              suppression_slack: int = -3,
              discard_boundaries: bool = True) -> np.ndarray:
    """
    Get a patch of a droplet from an image.
    ----------
    Parameters:
    image: np.ndarray:
        3D image data, where c represents channels, h is image height, and w is image width.
    center_x: float:
        x coordinate of the center of the droplet.
    center_y: float:
        y coordinate of the center of the droplet.
    radius: int:
        Radius of the droplet.
    buffer: int:
        Extra pixels of slack for extracting droplet patches.
    suppress_rest: bool:
        Whether to suppress pixels outside of the radius of the detected droplets.
    suppression_slack: int:
        Distance in pixels outside of the detected radius that is still considered part of the droplet.
    discard_boundaries: bool:
        Whether to exclude patches that are partially outside of the image boundaries.
    ----------
    Returns:
    ans: np.ndarray:
        A 3D array with the droplet patch.
    """
    s = image.shape
    assert len(s) == 3, 'axis length of image is not 2 or 3 (create_droplet_patches.py)'

    # We have channel, image_row and image_col as axes.
    window_dim = radius + buffer

    window_y = np.asarray((min(max(0, center_y - window_dim), s[1] - 1),
                           max(0, min(s[1], center_y + window_dim + 1))),
                          dtype=np.int32)

    window_x = np.asarray((min(max(0, center_x - window_dim), s[2] - 1),
                           max(0, min(s[2], center_x + window_dim + 1))),
                          dtype=np.int32)

    if ((window_y[1] - window_y[0] != 2 * window_dim + 1) or (
            window_x[1] - window_x[0] != 2 * window_dim + 1)) and discard_boundaries:

        return np.zeros((0, 0, 0))

    else:
        ans = np.zeros((s[0], 2 * window_dim + 1, 2 * window_dim + 1), dtype=np.uint16)
        target_rows = window_y - (center_y - window_dim)
        target_cols = window_x - (center_x - window_dim)

        ans[:, target_rows[0]: target_rows[1], target_cols[0]: target_cols[1]] = image[:,
                                                                                 window_y[0]: window_y[1],
                                                                                 window_x[0]: window_x[1]]

        if suppress_rest:
            mask = np.zeros(ans.shape[1:3], dtype=np.uint16)
            cv.circle(mask, np.asarray((window_dim, window_dim)), radius + suppression_slack, 1, -1)
            ans = ans * mask[None, :, :]
        return ans

# IMPORTANT! Default params are compatible with the current embeddings. Changing them will harm the performance.
def create_droplet_patches(image: np.ndarray, 
                           droplet_feature_table: pd.DataFrame, 
                           buffer: int = -2,
                           suppress_rest: bool = True, 
                           suppression_slack: int = -3,
                           discard_boundaries: bool = False) -> Dict:
    """
    Create a dataframe with droplet patches.
    ----------
    Parameters:
    image: np.ndarray:
        4D image data, where f represents frames, c represents channels, h is image height, and w is image width.
    droplet_feature_table: pd.DataFrame
        DataFrame with detected droplets.
    buffer: int
        Extra pixels of slack for extracting droplet patches.
    suppress_rest: bool
        Whether to suppress pixels outside of the radius of the detected droplets.
    suppression_slack: int
        Distance in pixels outside of the detected radius that is still considered part of the droplet.
    discard_boundaries: bool
        Whether to exclude patches that are partially outside of the image boundaries.
    ----------
    Returns:
    droplet_patches: Dict
        A dictionary with droplet_id, frame and patch for each droplet.
    """

    droplet_patches = {'droplet_id': [], 'frame': [], 'patch': []}

    for _, droplet in droplet_feature_table.iterrows():
        patch = get_patch(image[droplet['frame']], droplet['center_x'], droplet['center_y'],
                          droplet['radius'], buffer, suppress_rest, suppression_slack,
                          discard_boundaries)

        droplet_patches['droplet_id'].append(droplet['droplet_id'])
        droplet_patches['frame'].append(droplet['frame'])
        droplet_patches['patch'].append(patch)

    return droplet_patches


def create_and_save_droplet_patches(cfg: DictConfig, image_preprocessed_path: Path, image_feature_path: Path):
    """
    Creates droplet patches of the preprocessed cuts and stores them in a npy file.
    ----------
    Parameters:
    cfg: DictConfig:
        Global config.
    image_preprocessed_path: Path:
        Directory where preprocessed cuts are stored.
    image_feature_path: Path:
        Directory where .csv files with droplet features are stored.
    """
    if cfg.verbose:
        print("\n===================================================================")
        print("Cut Droplet Patches")
        print("===================================================================\n")
        print("Currently processing:")

    # Read droplet and cell tables from CSV files
    for filename in os.listdir(image_preprocessed_path):
        f = os.path.join(image_preprocessed_path, filename)
        # checking if it is a file
        if os.path.isfile(f) and filename.startswith("preprocessed_featextr_"):
            cut_file_name = filename

            if cfg.verbose:
                print(cut_file_name)

            preprocessed_cut_path = image_preprocessed_path / cut_file_name
            preprocessed_cut = np.load(str(preprocessed_cut_path))

            droplet_feature_file_name = preprocessed_cut_path.stem.replace("preprocessed_featextr_", "")
            droplet_feature_table = pd.read_csv(Path(image_feature_path / f'droplets_{droplet_feature_file_name}.csv'),
                                                index_col=False)

            droplet_patches = create_droplet_patches(preprocessed_cut, droplet_feature_table)
            np.save(str(image_feature_path / f'patches_{droplet_feature_file_name}.npy'), droplet_patches)
