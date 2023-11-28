from pathlib import Path

import numpy as np
import cv2 as cv
from tqdm.auto import tqdm
from skimage.filters import rank
from skimage.morphology import disk
from skimage.util import img_as_ubyte

from .raw_image_reader import get_image_as_ndarray
from .raw_image_reader import get_image_cut_as_ndarray


def preprocess_for_embeddings(cfg, image_path: str, pixel: int = -1) -> np.ndarray:
    """
    Preprocess the BF channel of an image
    ----------
    Parameters:
    path_to_image: str
        relative path to the image (image as .nd2 file)
    pixels: int
        num of pixels to cut out of the full image, if -1 the full image is taken
    ----------
    Returns:
    ndarray: 4d numpy array (uint16 so be careful)
        with the following axes: Frames, Channels ['BF', 'DAPI'], Y (rows) and X (cols),
        where the BF and DAPI channels have been processed with quantile and locally histogram equalized
    """
    image = get_image_as_ndarray(['BF', 'DAPI'], image_path, all_frames=True, all_channels=False, pixel=pixel)

    # For each frame, preprocess channels inplace
    image[:, 0, :, :] = np.uint16(2 ** 16 - (np.int32(image[:, 0, :, :]) + 1))

    for frame in tqdm(image, desc='BF and DAPI preprocessing'):
        # Brightfield preprocessing
        bf_chan = np.float64(frame[0, :, :])
        bf_chan_low = np.quantile(bf_chan, 0.1)
        bf_chan_high = np.quantile(bf_chan, 0.995)
        bf_chan = np.clip((bf_chan - bf_chan_low) / (bf_chan_high - bf_chan_low), 0.0, 1.0)

        img_medianblurred = np.float64(cv.medianBlur(np.uint8(bf_chan * 255), 2 * 50 + 1) / 255.0)
        img_mediansharpened = np.clip(bf_chan - img_medianblurred, 0.0, 1.0)
        equalized_bf = rank.equalize(img_as_ubyte(img_mediansharpened), footprint=disk(10)) / 255.0
        img_mediansharpened = img_mediansharpened * equalized_bf
        thresh = np.quantile(img_mediansharpened, 0.5)
        img_mediansharpened[img_mediansharpened > thresh] = bf_chan[img_mediansharpened > thresh]

        frame[0] = cv.GaussianBlur(np.uint16(img_mediansharpened * (2 ** 16 - 1)), (3, 3), 0)

        # DAPI preprocessing
        dapi_chan = np.float64(frame[1, :, :])
        dapi_chan_low = np.quantile(dapi_chan, 0.8)
        dapi_chan = np.clip((dapi_chan - dapi_chan_low) / ((2 ** 16 - 1) - dapi_chan_low), 0.0, 1.0)
        img_medianblurred = np.float64(cv.medianBlur(np.uint8(dapi_chan * 255), 2 * 20 + 1) / 255.0)
        img_mediansharpened = np.clip(dapi_chan - img_medianblurred, 0.0, 1.0)
        equalized_dapi = rank.equalize(img_as_ubyte(img_mediansharpened), footprint=disk(10)) / 255.0
        img_mediansharpened = img_mediansharpened * equalized_dapi
        thresh = np.quantile(img_mediansharpened, 0.8)
        img_mediansharpened[img_mediansharpened > thresh] = dapi_chan[img_mediansharpened > thresh]

        frame[1] = np.uint16(img_mediansharpened * (2 ** 16 - 1))

    image[np.isnan(image)] = 0.0
    return image

def preprocess_cut_for_embeddings(cfg, 
                            image_path: str, 
                            upper_left_corner: tuple,
                            pixel_dimensions: tuple,
                            pixel: int = -1) -> np.ndarray:
    """
    Preprocess the BF channel of an image
    ----------
    Parameters:
    path_to_image: str
        relative path to the image (image as .nd2 file)
    pixels: int
        num of pixels to cut out of the full image, if -1 the full image is taken
    ----------
    Returns:
    ndarray: 4d numpy array (uint16 so be careful)
        with the following axes: Frames, Channels ['BF', 'DAPI'], Y (rows) and X (cols),
        where the BF and DAPI channels have been processed with quantile and locally histogram equalized
    """
    image = get_image_cut_as_ndarray(cfg,
                                     ['BF', 'DAPI'], 
                                     image_path, 
                                     upper_left_corner,
                                     pixel_dimensions,
                                     all_frames=True, 
                                     all_channels=False, 
                                     frames = None,
                                     pixel=pixel)

    # For each frame, preprocess channels inplace
    image[:, 0, :, :] = np.uint16(2 ** 16 - (np.int32(image[:, 0, :, :]) + 1))

    # Disable progress bar if verbose is False
    disable = True 
    if cfg.verbose:
        disable = False

    for frame in tqdm(image, desc='BF and DAPI Preprocessing for Embeddings', disable=disable):
        # Brightfield preprocessing
        bf_chan = np.float64(frame[0, :, :])
        bf_chan_low = np.quantile(bf_chan, 0.1)
        bf_chan_high = np.quantile(bf_chan, 0.995)
        bf_chan = np.clip((bf_chan - bf_chan_low) / (bf_chan_high - bf_chan_low), 0.0, 1.0)

        img_medianblurred = np.float64(cv.medianBlur(np.uint8(bf_chan * 255), 2 * 50 + 1) / 255.0)
        img_mediansharpened = np.clip(bf_chan - img_medianblurred, 0.0, 1.0)
        equalized_bf = rank.equalize(img_as_ubyte(img_mediansharpened), footprint=disk(10)) / 255.0
        img_mediansharpened = img_mediansharpened * equalized_bf
        thresh = np.quantile(img_mediansharpened, 0.5)
        img_mediansharpened[img_mediansharpened > thresh] = bf_chan[img_mediansharpened > thresh]

        frame[0] = cv.GaussianBlur(np.uint16(img_mediansharpened * (2 ** 16 - 1)), (3, 3), 0)

        # DAPI preprocessing
        dapi_chan = np.float64(frame[1, :, :])
        dapi_chan_low = np.quantile(dapi_chan, 0.8)
        dapi_chan = np.clip((dapi_chan - dapi_chan_low) / ((2 ** 16 - 1) - dapi_chan_low), 0.0, 1.0)
        img_medianblurred = np.float64(cv.medianBlur(np.uint8(dapi_chan * 255), 2 * 20 + 1) / 255.0)
        img_mediansharpened = np.clip(dapi_chan - img_medianblurred, 0.0, 1.0)
        equalized_dapi = rank.equalize(img_as_ubyte(img_mediansharpened), footprint=disk(10)) / 255.0
        img_mediansharpened = img_mediansharpened * equalized_dapi
        thresh = np.quantile(img_mediansharpened, 0.8)
        img_mediansharpened[img_mediansharpened > thresh] = dapi_chan[img_mediansharpened > thresh]

        frame[1] = np.uint16(img_mediansharpened * (2 ** 16 - 1))

    image[np.isnan(image)] = 0.0
    return image

def raw_to_preprocessed_for_embeddings(cfg, raw_image_path: Path, image_name: str, PREPROCESSED_PATH: Path,
                                       pixel: int = -1) -> np.ndarray:
    """Preprocesses the raw .nd2 image for embedding creation and saves it as .npy file."""
    preprocessed_image = preprocess_for_embeddings(cfg, raw_image_path, pixel)
    path = Path(PREPROCESSED_PATH / f"preprocessed_featextr_bf_{image_name}.npy")
    np.save(path, preprocessed_image[:, 0, :, :])
    path = Path(PREPROCESSED_PATH / f"preprocessed_featextr_dapi_{image_name}.npy")
    np.save(path, preprocessed_image[:, 1, :, :])
    return preprocessed_image

def raw_cut_to_preprocessed_for_embeddings(cfg, 
                                           raw_image_path: Path, 
                                           upper_left_corner: tuple,
                                           pixel_dimensions: tuple,
                                           image_name: str, 
                                           preprocessed_path: Path,
                                           pixel: int = -1) -> np.ndarray:
    """Preprocesses the raw .nd2 image for embedding creation and saves it as .npy file."""
    preprocessed_image = preprocess_cut_for_embeddings(cfg, raw_image_path, upper_left_corner, pixel_dimensions, pixel=pixel)
    path = Path(preprocessed_path / f"preprocessed_featextr_bf_{image_name}.npy")
    np.save(path, preprocessed_image[:, 0, :, :])
    path = Path(preprocessed_path / f"preprocessed_featextr_dapi_{image_name}.npy")
    np.save(path, preprocessed_image[:, 1, :, :])
    return preprocessed_image
