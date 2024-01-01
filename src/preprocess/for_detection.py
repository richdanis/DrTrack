# Types
from pathlib import Path
from typing import Tuple

# Preprocessing libraries
import numpy as np
import cv2 as cv
from skimage.filters import rank
from skimage.morphology import disk
from skimage.util import img_as_ubyte

# Local imports
from .raw_image_reader import get_image_cut_as_ndarray

# Progress
from tqdm.auto import tqdm

def preprocess_cut_for_detection(cfg,
                                 image_path: Path,
                                 upper_left_corner: Tuple[int, int],
                                 pixel_dimensions: Tuple[int, int],
                                 pixel: int = -1) -> np.ndarray:
    """
    Preprocess the BF channel of a cut of an nd2 image
    ----------
    Parameters:
    cfg: Config
    image_path: Path
        relative path to the image (image as .nd2 file)
    upper_left_corner: Tuple[int, int]
        upper left corner of the cut (y, x)
    pixel_dimensions: Tuple[int, int]
        dimensions of the cut (y, x)
    pixels: Optional[int]
        num of pixels to cut out of the full image, if -1 the full image is taken
    ----------
    Returns:
    ndarray: 4d numpy array (uint16 so be careful)
        with the following axes: Frames, Channels ['BF', 'DAPI'], Y (rows) and X (cols),
        where the BF channel has been processed with quantile and locally histogram equalized
    """
    image = get_image_cut_as_ndarray(cfg,
                                     ['BF', 'DAPI'],
                                     image_path,
                                     upper_left_corner,
                                     pixel_dimensions,
                                     all_frames=True,
                                     all_channels=False,
                                     frames=None,
                                     pixel=pixel)

    # For each frame, preprocess channels inplace
    image[:, 0, :, :] = np.uint16(2 ** 16 - (np.int32(image[:, 0, :, :]) + 1))

    for frame in tqdm(image, desc='BF Preprocessing for Detection', disable=cfg.tqdm_disable):
        # Brightfield preprocessing
        bf_chan = np.float64(frame[0, :, :])
        bf_chan_low = np.quantile(bf_chan, 0.1)
        bf_chan_high = np.quantile(bf_chan, 0.995)
        bf_chan = np.clip((bf_chan - bf_chan_low) / (bf_chan_high - bf_chan_low), 0.0, 1.0)

        pullback_min = bf_chan.min()
        pullback_max = bf_chan.max()
        bf_pullback = (bf_chan - pullback_min) / (pullback_max - pullback_min)

        bf_pullback = np.clip((bf_pullback - np.quantile(bf_pullback, 0.5)) / (1.0 - np.quantile(bf_pullback, 0.5)),
                              0.0, 1.0)

        equalized = rank.equalize(img_as_ubyte(bf_pullback), footprint=disk(10)) / 255.0

        bf_pullback = bf_pullback * equalized

        smoothed = cv.GaussianBlur(bf_pullback, (3, 3), 0)
        frame[0, :, :] = np.uint16(smoothed * (2 ** 16 - 1))

    return image


def raw_cut_to_preprocessed_for_detection(cfg,
                                          raw_image_path: Path,
                                          upper_left_corner: Tuple[int, int],
                                          pixel_dimensions: Tuple[int, int],
                                          image_name: str,
                                          preprocessed_path: Path,
                                          pixel: int = -1) -> np.ndarray:
    """
    Preprocess an image for droplet detection and save as .npy array in the preprocessing data directory.
    ----------
    Parameters:
    cfg: DictConfig
        Global config.
    raw_image_path: Path
        Relative path to the image (image as .nd2 file)
    upper_left_corner: Tuple[int, int]
        Upper left corner of the cut (y, x)
    pixel_dimensions: Tuple[int, int]
        Dimensions of the cut (y, x)
    image_name: str
        Name of the image
    preprocessed_path: Path
        Path to save the preprocessed image
    pixel: Optional[int]
        Num of pixels to cut out of the full image, if -1 the full image is taken
    ----------
    Returns:
    None
    """
    preprocessed_image = preprocess_cut_for_detection(cfg, raw_image_path, upper_left_corner, pixel_dimensions,
                                                      pixel=pixel)
    file_path = Path(preprocessed_path / f"preprocessed_drpdtc_{image_name}.npy")
    np.save(file_path, preprocessed_image)
