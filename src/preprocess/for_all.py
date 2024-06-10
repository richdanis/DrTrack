# Types
from typing import Optional, Tuple
from pathlib import Path
from omegaconf import DictConfig

# Handling raw files
import nd2

# Local imports
from .for_embeddings import raw_cut_to_preprocessed_for_embeddings
from .for_detection import raw_cut_to_preprocessed_for_detection


def preprocess_cut_and_store(cfg: DictConfig, 
                             raw_image_path: Path, 
                             upper_left_corner: Tuple[int, int],
                             pixel_dimensions: Tuple[int, int],
                             image_name: str, 
                             PREPROCESSED_PATH: Path, 
                             pixel: Optional[int] = -1):
    """
    Preprocess the a single cut for embedding creation and droplet detection and save as .npy arrays.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary.
    raw_image_path : Path
        Path to the raw image file.
    upper_left_corner : tuple
        Tuple representing the coordinates of the upper left corner of the region of interest.
    pixel_dimensions : tuple
        Tuple representing the dimensions of the region of interest in pixels.
    image_name : str
        Name of the image.
    PREPROCESSED_PATH : Path
        Path to save the preprocessed .npy arrays.
    pixel : int, optional
        The pixel value to use for preprocessing. Default is -1.

    Returns
    -------
    None
    """
    raw_cut_to_preprocessed_for_detection(cfg, raw_image_path, upper_left_corner, pixel_dimensions, image_name,
                                          PREPROCESSED_PATH, pixel=pixel)
    raw_cut_to_preprocessed_for_embeddings(cfg, raw_image_path, upper_left_corner, pixel_dimensions, image_name,
                                           PREPROCESSED_PATH, pixel=pixel)


def preprocess_cuts_and_store_all(cfg, RAW_PATH: Path, PREPROCESSED_PATH: Path):
    """
    Preprocess the .nd2 images separated into cuts and save as .npy arrays.

    Parameters
    ----------
    cfg : object
        Configuration object containing preprocessing parameters.
    RAW_PATH : Path
        Path to the raw .nd2 image file.
    PREPROCESSED_PATH : Path
        Path to store the preprocessed .npy arrays.

    Returns
    -------
    None
    """
    # Create paths
    raw_image_path = Path(RAW_PATH / cfg.raw_image)

    # Preprocess the image cuts if required
    # Open nd2 file and check sizes
    f = nd2.ND2File(raw_image_path)
    size_y = f.sizes['Y']
    size_x = f.sizes['X']

    # Close nd2 file
    f.close()

    # Get desired numbers of cuts in each dimension
    cuts_y = cfg.preprocess.cuts_y
    cuts_x = cfg.preprocess.cuts_x

    # Cut image into smaller images
    cut_size_y = size_y // cuts_y
    cut_size_x = size_x // cuts_x
    cut_pixel_dimensions = (cut_size_y, cut_size_x)

    if cfg.verbose:
        print("===================================================================")
        print("Preprocess the Dataset for Detection and Embedding Creation")
        print("===================================================================\n")
        print(f'Nr frames: {f.sizes["T"]}')
        print(f'Nr channels: {f.sizes["C"]}')
        print(f'Image dimensions: {cuts_y}x{cuts_x}')
        print(f'Number of cuts: {cuts_y * cuts_x}')
        print(f'Cut size: {cut_size_y}x{cut_size_x}\n')
        print(f'Currently Processing:')

    # Process and store each cut
    cut_image_names = []
    for i in range(cuts_y):
        for j in range(cuts_x):
            # Naming convention: image_name_y[upper_left]_x[upper_left]
            image_name_curr = f'y{i * cut_size_y}_x{j * cut_size_x}'
            upper_left_corner = (i * cut_size_y, j * cut_size_x)

            # Progress
            if cfg.verbose:
                print("-------------------------------------------------------------------")
                print(image_name_curr)

            preprocess_cut_and_store(cfg, raw_image_path, upper_left_corner, cut_pixel_dimensions, image_name_curr,
                                     PREPROCESSED_PATH, pixel=cfg.preprocess.pixel)

            # Save image names in list
            cut_image_names.append(image_name_curr)
