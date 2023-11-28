import os
import re
from pathlib import Path
import toml
import pandas as pd
import random
import logging
import torch
import numpy as np
from .data_creation import droplets_and_cells

def detect_and_store_cut(cfg,
                         cut_file_name,
                         FEATURE_PATH, 
                         PREPROCESSED_PATH) -> None:

    # Retrieve image
    preprocessed_cut_path = Path(PREPROCESSED_PATH / cut_file_name)
    preprocessed_cut = np.load(preprocessed_cut_path)

    droplet_feature_file_name = preprocessed_cut_path.stem.replace("preprocessed_drpdtc_", "")

    droplet_feature_path = Path(FEATURE_PATH / f"droplets_{droplet_feature_file_name}.csv")
    #cell_feature_path = Path(FEATURE_PATH / f"cells_{image_name}.csv")
    droplets_and_cells.generate_output_from_ndarray(cfg,
                                                    preprocessed_cut, 
                                                    droplet_feature_path,
                                                    True, "", False, 
                                                    radius_min = cfg.detect_droplets.radius_min, 
                                                    radius_max = cfg.detect_droplets.radius_max)

def detect_and_store_all(cfg, image_preprocessed_path, image_feature_path):
    """ 
    Detect droplets and cells in all preprocessed cuts of an image and store them in a csv file
    """
    if cfg.verbose == True:
        print("Currently Processing:")
    for filename in os.listdir(image_preprocessed_path):
        f = os.path.join(image_preprocessed_path, filename)
        # checking if it is a file
        if os.path.isfile(f) and filename.startswith("preprocessed_drpdtc_"):
            cut_file_name = filename
            detect_and_store_cut(cfg, cut_file_name, image_feature_path, image_preprocessed_path)

            if cfg.verbose == True:
                print(cut_file_name)