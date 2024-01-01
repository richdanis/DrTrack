# os
import os

# Types
from pathlib import Path

# Handling arrays
import numpy as np

# Local imports
from .data_creation import droplets_and_cells

def detect_and_store_cut(cfg,
                         cut_file_name: Path,
                         FEATURE_PATH: Path, 
                         PREPROCESSED_PATH: Path) -> None:
    """
    Detect droplets in a preprocessed cut of the image and store the results.

    Parameters
    ----------
    cfg : object
        The configuration object.
    cut_file_name : Path
        The name of the cut file preprocessed for droplet detection.
    FEATURE_PATH : Path
        The path where created features should be stored.
    PREPROCESSED_PATH : Path
        The path to the preprocessed files.

    Returns
    -------
    None
    """

    # Retrieve image
    preprocessed_cut_path = Path(PREPROCESSED_PATH / cut_file_name)
    preprocessed_cut = np.load(preprocessed_cut_path)
    droplet_feature_file_name = preprocessed_cut_path.stem.replace("preprocessed_drpdtc_", "")
    droplet_feature_path = Path(FEATURE_PATH / f"droplets_{droplet_feature_file_name}.csv")

    # Detect droplets and store results
    droplets_and_cells.detect_droplets_and_cells_and_store(cfg,
                                                    input_image=preprocessed_cut, 
                                                    output_string_droplets=droplet_feature_path,
                                                    refine=True,
                                                    radius_min=cfg.detect_droplets.radius_min, 
                                                    radius_max=cfg.detect_droplets.radius_max)


def detect_and_store_all(cfg, image_preprocessed_path: Path, image_feature_path: Path):
    """
    Detect droplets and cells in all preprocessed cuts of an image and store them in a csv file.

    Parameters
    ----------
    cfg : object
        The configuration object containing various settings.
    image_preprocessed_path : Path
        The path to the directory containing preprocessed image cuts.
    image_feature_path : Path
        The path to the directory where the detected droplets and cells will be stored.
    """
    # Progress
    if cfg.verbose:
        print("\n===================================================================")
        print("Detect Droplet Locations and Estimate Numbers of Cells")
        print("===================================================================\n")
      
    if cfg.verbose == True:
        print("Currently Processing:")

    # Detect droplets and cells in all preprocessed cuts of the image
    for filename in os.listdir(image_preprocessed_path):
        f = os.path.join(image_preprocessed_path, filename)

        # checking if it is the correct type of file
        if os.path.isfile(f) and filename.startswith("preprocessed_drpdtc_"):
            cut_file_name = filename

            if cfg.verbose == True:
                print(cut_file_name)
                
            detect_and_store_cut(cfg, cut_file_name, image_feature_path, image_preprocessed_path)
