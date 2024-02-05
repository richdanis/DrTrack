# Types
from pathlib import Path

# Computer vision for circle detection
import cv2 as cv

# Handling arrays
import numpy as np

# Data handling
import pandas as pd

# Progress
from tqdm.auto import tqdm

# Local imports
from . import manual_circle_hough, cell_detector


# The input image should be an ndarray with shape (f,c,h,w) where f = frames, c = channels, h = height and w = width of the image.
# IMPORTANT: Datatype should be uint16 just as with the raw images and BF and DAPI must be channels Nr 0 and 1 respectively
def detect_droplets_and_cells_and_store(cfg,
                                 input_image: np.ndarray,
                                 output_string_droplets: Path,
                                 refine: bool,
                                 radius_min: int = 12,
                                 radius_max: int = 25) -> None:
    """
    Detect droplets and estimate the number of cells and store the results.

    Parameters
    ----------
    cfg : Configuration object
        The configuration object.
    input_image : np.ndarray
        The input image as a numpy ndarray.
    output_string_droplets : Path
        The path to the output file for droplets.
    refine : bool
        A boolean flag indicating whether to refine the circles.
    radius_min : int, optional
        The minimum radius for circle detection. Default is 12.
    radius_max : int, optional
        The maximum radius for circle detection. Default is 25.

    Returns
    -------
    None
        This function does not return anything.
    """
    # Check dimensions of input image
    nr_frames = input_image.shape[0]
    nr_channels = input_image.shape[1]

    # Detect droplets and cells in all frames
    droplets = []
    cells_dict = []
    for frame_nr in tqdm(range(nr_frames), disable=cfg.tqdm_disable):
        # Extract channels
        dapi_channel = input_image[frame_nr, 1, :, :]
        bf_channel = input_image[frame_nr, 0, :, :]
        visualization_channel = np.zeros(bf_channel.shape, dtype=np.float32)

        # Detect circles
        circles_in_frame = manual_circle_hough.manual_circle_hough(bf_channel, refine, bf_is_inverted=True,
                                                                   radius_min=radius_min, radius_max=radius_max)

        # Detect cells
        cells_mask, cells_intensities, cells_persistencies = cell_detector.cell_detector(cfg, dapi_channel, bf_channel,
                                                                                         circles_in_frame)

        # Use intensity and persistence to estimate number of cells in each droplet
        intensities_vector = cells_intensities[cells_mask == 1.0]
        persistence_vector = cells_persistencies[cells_mask == 1.0]

        intens_thresh = np.quantile(intensities_vector, 0.2)
        presis_thresh = np.quantile(persistence_vector, 0.2)
        visualization_channel = cv.morphologyEx(cells_mask, cv.MORPH_DILATE, np.ones((3, 3)))

        cell_id_counter = 0

        # Loop over all detected circles
        for id, circ in enumerate(circles_in_frame):
            # Extract information about the circle
            center = np.asarray([circ[0], circ[1]])
            radius = circ[2]
            patch_x = (max(int(center[0]) - radius - 2, 0), min(int(center[0]) + radius + 2, cells_mask.shape[0] - 1))
            patch_y = (max(int(center[1]) - radius - 2, 0), min(int(center[1]) + radius + 2, cells_mask.shape[1] - 1))
            local_cells_mask = cells_mask[patch_x[0]: patch_x[1], patch_y[0]: patch_y[1]]
            local_cells_intens = cells_intensities[patch_x[0]: patch_x[1], patch_y[0]: patch_y[1]]
            local_cells_pers = cells_persistencies[patch_x[0]: patch_x[1], patch_y[0]: patch_y[1]]

            local_mask = np.zeros(local_cells_mask.shape)
            center_in_patch = center - np.asarray(
                [max(int(center[0]) - radius - 2, 0), max(int(center[1]) - radius - 2, 0)])
            cv.circle(local_mask, np.flip(center_in_patch), radius, 1.0, -1)
            local_cells_mask = local_cells_mask * local_mask
            local_cells_intens = local_cells_intens * local_mask
            local_cells_pers = local_cells_pers * local_mask

            # Estimate number of cells in droplet
            nr_cells_estimated = np.sum(
                np.logical_and((local_cells_pers > presis_thresh), (local_cells_intens > intens_thresh)))
            
            cv.circle(visualization_channel, np.flip(center), radius, 1.0, 1)

            # Store droplet and cell information
            droplets.append(
                {"droplet_id": id, "frame": frame_nr, "center_y": circ[0], "center_x": circ[1], "radius": circ[2],
                 "nr_cells": nr_cells_estimated})
            cell_coords = np.transpose(np.asarray(np.where(local_cells_mask != 0.0)))
            for coord in cell_coords:
                global_center = coord + np.asarray(
                    [max(int(center[0]) - radius - 2, 0), max(int(center[1]) - radius - 2, 0)])
                cells_dict.append({"cell_id": cell_id_counter,
                                   "droplet_id": id,
                                   "frame": frame_nr,
                                   "center_y": global_center[0],
                                   "center_x": global_center[1],
                                   "intensity_score": local_cells_intens[coord[0], coord[1]],
                                   "persistence_score": local_cells_pers[coord[0], coord[1]]})
                cell_id_counter = cell_id_counter + 1

    # Store droplet and cell information in csv files
    droplet_df = pd.DataFrame(droplets)

    # Drop droplets with radius less than minimal radius indicated in config file
    droplet_df = droplet_df[droplet_df['radius'] >= cfg.detect_droplets.radius_min]

    droplet_df.to_csv(output_string_droplets, index=False)
