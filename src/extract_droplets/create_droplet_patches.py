import sys
from pathlib import Path
import os

import numpy as np
import cv2 as cv
import pandas as pd

sys.path.append('./src/')
from preprocess.raw_image_reader import get_image_cut_as_ndarray


def get_patch(image, center_y, center_x, radius, buffer=3, suppress_rest=True, suppression_slack=1,
              discard_boundaries=True):
    s = image.shape
    assert (len(s) == 2 or len(s) == 3), 'axis length of image is not 2 or 3 (create_droplet_patches.py)'
    if len(s) == 3:
        # We are in the case where we have channel, image_row and image_col as axes.
        window_dim = radius + buffer
        window_y = np.asarray((max(0, center_y - window_dim), min(s[1], center_y + window_dim + 1)), dtype=np.int32)

        window_y = np.asarray((min(max(0, center_y - window_dim), s[1] - 1),
                               max(0, min(s[1], center_y + window_dim + 1))),
                              dtype=np.int32)

        window_x = np.asarray((max(0, center_x - window_dim), min(s[2], center_x + window_dim + 1)), dtype=np.int32)

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

    elif len(s) == 2:
        window_dim = radius + buffer
        window_y = np.asarray((max(0, center_y - window_dim), min(s[0], center_y + window_dim + 1)), dtype=np.int32)
        window_x = np.asarray((max(0, center_x - window_dim), min(s[1], center_x + window_dim + 1)), dtype=np.int32)

        window_y = np.asarray((min(max(0, center_y - window_dim), s[0] - 1),
                               max(0, min(s[0], center_y + window_dim + 1))),
                              dtype=np.int32)

        window_x = np.asarray((min(max(0, center_x - window_dim), s[1] - 1),
                               max(0, min(s[1], center_x + window_dim + 1))),
                              dtype=np.int32)

        if ((window_y[1] - window_y[0] != 2 * window_dim + 1) or (
                window_x[1] - window_x[0] != 2 * window_dim + 1)) and discard_boundaries:
            return np.zeros((0, 0, 0))
        else:
            ans = np.zeros((2 * window_dim + 1, 2 * window_dim + 1), dtype=np.uint16)
            target_rows = window_y - (center_y - window_dim)
            target_cols = window_x - (center_x - window_dim)
            ans[target_rows[0]: target_rows[1], target_cols[0]: target_cols[1]] = image[window_y[0]: window_y[1],
                                                                                  window_x[0]: window_x[1]]
            if suppress_rest:
                mask = np.zeros(ans.shape, dtype=np.uint16)
                cv.circle(mask, np.asarray((window_dim, window_dim)), radius + suppression_slack, 1, -1)
                ans = ans * mask
            return ans


# frames is the list of frames (integer index) you want to retreive
# channels is the list of channels (strings) you want to retreive.
#   'BF' is the identifier for the brightfield images
#   'DAPI' is the identifier for the DAPI channel
# image_path is the full path to the nd2 image, suffix included
# droplet_table_path is the path to the csv table with the detected droplets
# allFrames indicates whether we want to retreive all frames.
# allChannels indicates whether we want to retrieve all channels.
# buffer is the number of extra pixels of slack that we want to have when cutting out the droplets. So the dimension of the returned patches is 2 * (slack + radius) + 1.
# suppress_rest indicates whether we want to suppress the pixels outside of the radius
# suppression_slack is the distance in pixels outside of teh detected radius, that we still consider to be part of the droplet and which we dont suppress.
#   So, everything that is farther away than radius + suppression_slack from the droplet center, gets suppressed
# discard_boundaries indicates whether to not cut out patches that are not 100% included in the image. If set to false, regions of the patch that exceed image boundaries are filled with zeros.
#   If set to true, droplets whose image patches are not contained in the image get a patch of 0x0 pixels.
# returns a list with one element for each frame. Each element is again a list of dicts / dataframes (not sure)
#   which contains all the data about the droplet plus a 'patch' which is the image patch around the droplet with according channels

# Example use:
"""
    image_path = 'raw_images/smallMovement1.nd2'
    table_path = 'finished_outputs/smallMovement1_droplets.csv'
    dataset = create_dataset([0], ['BF'], image_path, table_path, allFrames = True, allChannels = True)
    print(len(dataset))
    print(dataset[0][0])
    print(dataset[0][0]['patch'].shape)
    # Iterate over frames
    for fr in dataset:
        # Upscale one patch and all its channels at once
        upscaled = resize_patch(fr[0]['patch'], 100)
        print(upscaled.shape)
        # Display channels
        for ch in upscaled:
            cv.imshow("test", ch)
            cv.waitKey(0)
"""


def create_dataset(channels, image_path, droplet_table_path, allFrames=True, allChannels=False, buffer=3,
                   suppress_rest=True, suppression_slack=1, discard_boundaries=False):
    droplet_table = pd.read_csv(droplet_table_path, index_col=False)
    raw_images = get_image_cut_as_ndarray(channels, image_path, allFrames, allChannels)

    if allFrames:
        new_frames = range(raw_images.shape[0])

    ans = []
    for i, j in enumerate(new_frames):
        droplets_in_frame = droplet_table[droplet_table['frame'] == j]
        image_frame = raw_images[i, :, :, :]
        frame_ans = []
        for idx, droplet in droplets_in_frame.iterrows():
            tmp_ans = droplet
            tmp_ans['patch'] = get_patch(image_frame, droplet['center_y'], droplet['center_x'], droplet['radius'],
                                         buffer, suppress_rest, suppression_slack, discard_boundaries)
            frame_ans.append(tmp_ans)
        ans.append(frame_ans)
    return ans


# Will return a new patch which is the old patch down or up scaled to have height and width 'diameter'. Only square input-patches are allowed.
# Will assume that the last two axes of the input patch are y and x. Channels supported.
# Does both up and down scaling

# Example use:
"""
    image_path = 'raw_images/smallMovement1.nd2'
    table_path = 'finished_outputs/smallMovement1_droplets.csv'
    dataset = create_dataset(['BF'], image_path, table_path, allFrames = True, allChannels = True)
    print(len(dataset))
    print(dataset[0][0])
    print(dataset[0][0]['patch'].shape)
    # Iterate over frames
    for fr in dataset:
        # Upscale one patch and all its channels at once
        upscaled = resize_patch(fr[0]['patch'], 100)
        print(upscaled.shape)
        # Display channels
        for ch in upscaled:
            cv.imshow("test", ch)
            cv.waitKey(0)
"""


def resize_patch(patch, diameter):
    s = np.asarray(patch.shape)

    numaxes = len(s)
    dims = (s[-2], s[-1])

    assert (dims[0] == dims[1])
    are_we_upscaling = (dims[0] <= diameter)

    if numaxes == 2:
        if are_we_upscaling:
            return cv.resize(patch, (diameter, diameter), interpolation=cv.INTER_CUBIC)
        else:
            return cv.resize(patch, (diameter, diameter), interpolation=cv.INTER_AREA)
    else:
        target_dim = np.copy(s)
        target_dim[-2] = diameter
        target_dim[-1] = diameter
        ans = np.zeros(target_dim, dtype=patch.dtype)

        for i in range(np.prod(np.asarray(s[0: -2]))):
            idx = np.unravel_index(i, s[0: -2])

            if are_we_upscaling:
                ans[idx] = cv.resize(patch[idx], (diameter, diameter), interpolation=cv.INTER_CUBIC)
            else:
                ans[idx] = cv.resize(patch[idx], (diameter, diameter), interpolation=cv.INTER_AREA)
        return ans


def create_droplet_patches(image, droplet_feature_table, buffer=3, suppress_rest=True, suppression_slack=1,
                                              discard_boundaries=False, get_patches=True):
    """
    Create a dataset that combines droplet and cell information from input images and tables.
    ----------
    Parameters:
    image: np.ndarray:
        4D image data, where f represents frames, c represents channels, h is image height, and w is image width.
    droplet_table_path: str
        Path to the CSV table with detected droplets.
    buffer: int
        Extra pixels of slack for extracting droplet patches.
    suppress_rest: bool
        Whether to suppress pixels outside of the radius of the detected droplets.
    suppression_slack: int
        Distance in pixels outside of the detected radius that is still considered part of the droplet.
    discard_boundaries: bool
        Whether to exclude patches that are partially outside of the image boundaries.
    get_patches: bool
        If True, omit creating patches; return data without patches.
    ----------
    Returns:
    ans: list[list[pd.Series]]
        A list with one element for each frame. Each element is a list of dictionaries (or dataframes) containing data about droplets and their associated patches
        (if not omitted). The 'cell_signals' field in each dictionary contains data about cell signals in the droplet.
    """

    droplet_patches = []

    if get_patches:

        for _, droplet in droplet_feature_table.iterrows():
            patch = get_patch(image[droplet['frame']], droplet['center_x'], droplet['center_y'],
                              droplet['radius'], buffer, suppress_rest, suppression_slack,
                              discard_boundaries)

            droplet_patches.append((droplet['droplet_id'], patch))

    droplet_patches_df = pd.DataFrame(droplet_patches, columns=['droplet_id', 'patch'])
    return droplet_patches_df

def create_and_save_droplet_patches(cfg, image_preprocessed_path, image_feature_path):
    # Read droplet and cell tables from CSV files
    for filename in os.listdir(image_preprocessed_path):
        f = os.path.join(image_preprocessed_path, filename)
        # checking if it is a file
        if os.path.isfile(f) and filename.startswith("preprocessed_featextr_bf_"):
            cut_file_name = filename

            preprocessed_cut_path = Path(image_preprocessed_path / cut_file_name)
            preprocessed_cut = np.load(preprocessed_cut_path)

            droplet_feature_file_name = preprocessed_cut_path.stem.replace("preprocessed_featextr_bf_", "droplet_")
            droplet_feature_table = pd.read_csv(image_feature_path / droplet_feature_file_name, index_col=False)

            droplet_patches_df = create_droplet_patches(preprocessed_cut, droplet_feature_table)
            droplet_patch_file_name = preprocessed_cut_path.stem.replace("preprocessed_featextr_bf_", "patches_")
            np.save(droplet_patch_file_name, droplet_patches_df)
