import os
import time
import argparse
from pathlib import Path
import nd2
import numpy as np
import pandas as pd

import jax

jax.devices("cpu")[0]

from preprocess.for_detection import raw_to_preprocessed_for_detection
from preprocess.for_embeddings import raw_to_preprocessed_for_embeddings
from preprocess.for_detection import raw_cut_to_preprocessed_for_detection
from preprocess.for_embeddings import raw_cut_to_preprocessed_for_embeddings
from preprocess.for_all import preprocess_cuts_and_store_all
from detect_droplets.detect_and_store import detect_and_store_all
from extract_droplets.create_droplet_patches import create_and_save_droplet_patches
from extract_visual_embeddings.create_visual_embeddings import create_and_save_droplet_embeddings
from track.ot import OptimalTransport
from generate_results.get_trajectories import compute_and_store_results_all

#from utils.globals import *

import hydra
from omegaconf import DictConfig, OmegaConf


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def setup_directories(cfg):
    # Create directories if they do not exist
    create_dir(Path(cfg.data_path))
    create_dir(Path(cfg.models_path))
    create_dir(Path(cfg.data_path) / Path(cfg.simulated_dir))
    create_dir(Path(cfg.data_path) / Path(cfg.preprocessed_dir))
    create_dir(Path(cfg.data_path) / Path(cfg.feature_dir))
    create_dir(Path(cfg.data_path) / Path(cfg.ot_dir))
    create_dir(Path(cfg.data_path) / Path(cfg.results_dir))

@hydra.main(config_path="conf", config_name="config_evaluate_tracking", version_base=None)
def main(cfg: DictConfig):
    # Setup directories
    RAW_PATH = Path(cfg.data_path) / Path(cfg.simulated_dir)
    PREPROCESSED_PATH = Path(cfg.data_path) / Path(cfg.preprocessed_dir)
    FEATURE_PATH = Path(cfg.data_path) / Path(cfg.feature_dir)
    OT_PATH = Path(cfg.data_path) / Path(cfg.ot_dir)
    RESULTS_PATH = Path(cfg.data_path) / Path(cfg.results_dir)
    setup_directories(cfg)

    # Start timer
    start_time = time.time()

if __name__ == '__main__':
    main()
