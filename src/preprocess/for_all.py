from .for_embeddings import raw_cut_to_preprocessed_for_embeddings  # , raw_to_preprocessed_for_embeddings
from .for_detection import raw_cut_to_preprocessed_for_detection  # , raw_to_preprocessed_for_detection
from typing import Optional
from pathlib import Path
import nd2


def preprocess_cut_and_store(cfg, raw_image_path: Path, upper_left_corner: tuple, pixel_dimensions: tuple,
                             image_name: str, PREPROCESSED_PATH: str, pixel: Optional[int] = -1):
    """Preprocess the .nd2 images and save as .npy arrays."""
    raw_cut_to_preprocessed_for_detection(cfg, raw_image_path, upper_left_corner, pixel_dimensions, image_name,
                                          PREPROCESSED_PATH, pixel=pixel)
    raw_cut_to_preprocessed_for_embeddings(cfg, raw_image_path, upper_left_corner, pixel_dimensions, image_name,
                                           PREPROCESSED_PATH, pixel=pixel)


def preprocess_cuts_and_store_all(cfg, RAW_PATH: Path, preprocessed_path: Path, image_name: str):
    """
    Preprocess the .nd2 images separated into cuts and save as .npy arrays.
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
            image_name_curr = f'{image_name}_y{i * cut_size_y}_x{j * cut_size_x}'
            upper_left_corner = (i * cut_size_y, j * cut_size_x)

            # Progress
            if cfg.verbose:
                print(image_name_curr)

            preprocess_cut_and_store(cfg, raw_image_path, upper_left_corner, cut_pixel_dimensions, image_name_curr,
                                     preprocessed_path, pixel=cfg.preprocess.pixel)

            # Save image names in list
            cut_image_names.append(image_name_curr)
