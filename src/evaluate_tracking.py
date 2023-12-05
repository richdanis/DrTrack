import os
import time
import argparse
from pathlib import Path
import nd2
import numpy as np
import pandas as pd
import shutil
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
from evaluate.preprocess_simulated import SimulatedData
from evaluate.get_scores import OtEvaluation
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

@hydra.main(config_path="../conf", config_name="config_evaluate_tracking", version_base=None)
def main(cfg: DictConfig):
    # Setup directories
    SIMULATED_PATH = Path(cfg.data_path) / Path(cfg.simulated_dir)
    PREPROCESSED_PATH = Path(cfg.data_path) / Path(cfg.preprocessed_dir)
    FEATURE_PATH = Path(cfg.data_path) / Path(cfg.feature_dir)
    OT_PATH = Path(cfg.data_path) / Path(cfg.ot_dir)
    RESULTS_PATH = Path(cfg.data_path) / Path(cfg.results_dir)
    setup_directories(cfg)


    ### PREPROCESSING ###
    # Preprocess simulated data
    image_simulated = Path(SIMULATED_PATH / cfg.simulated_image)
    image_preprocessed_path = Path(PREPROCESSED_PATH / cfg.experiment_name)
    image_feature_path = Path(FEATURE_PATH / cfg.experiment_name)
    create_dir(image_preprocessed_path)
    create_dir(image_feature_path)

    if not cfg.skip_preprocessing:
        sim_data = SimulatedData(cfg, image_simulated, image_feature_path)
        sim_data.create_and_store_position_dfs()


    ### VISUAL EMBEDDING EXTRACTION ###
    # Check conf/extract_features.yaml for settings

    if not cfg.skip_visual_embedding_extraction:
        # Create paths if they do not exist
        create_dir(image_feature_path)

        # Copy paired patches to data_path/feature_dir
        paired_patches_path = FEATURE_PATH / Path(cfg.paired_patches)
        new_name = "patches_" + cfg.experiment_name + ".npy"
        new_path = image_feature_path / Path(new_name)

        # Copy to required location if it does not exist
        if not os.path.exists(new_path):
            shutil.copyfile(paired_patches_path, new_path)

        # Create embeddings using configured model
        create_and_save_droplet_embeddings(cfg, image_feature_path)

    
    ### TRACKING ###
    image_ot_path = Path(OT_PATH / cfg.experiment_name)
    if not cfg.skip_tracking:
        # Create paths if they do not exist
        create_dir(image_ot_path)

        test_features = np.random.rand(3, 2, 3)
        ot = OptimalTransport(cfg)
        ot.compute_and_store_ot_matrices_all(image_feature_path, image_ot_path)


    ### GENERATING RESULTS (Trajectories and Scores) ###
    image_results_path = Path(RESULTS_PATH / cfg.experiment_name)

    if not cfg.skip_results_generation:
        # Create paths if they do not exist
        create_dir(image_results_path)

        if not cfg.skip_trajectory_generation:
            compute_and_store_results_all(cfg, image_ot_path, image_results_path, image_feature_path)

        if not cfg.skip_scoring:
            ot_evaluation = OtEvaluation(cfg, image_simulated, image_ot_path, image_results_path)
            ot_evaluation.compute_and_store_scores()

if __name__ == '__main__':
    main()
