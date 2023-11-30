import os
import time
from pathlib import Path

import numpy as np
import hydra
from omegaconf import DictConfig
import jax

jax.devices("cpu")[0]

from preprocess.for_all import preprocess_cuts_and_store_all
from detect_droplets.detect_and_store import detect_and_store_all
from extract_droplets.create_droplet_patches import create_and_save_droplet_patches
from extract_visual_embeddings.create_visual_embeddings import create_and_save_droplet_embeddings
from track.ot import OptimalTransport
from generate_results.get_trajectories import compute_and_store_results_all


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def setup_directories(cfg):
    # Create directories if they do not exist
    create_dir(Path(cfg.data_path))
    create_dir(Path(cfg.data_path) / Path(cfg.raw_dir))
    create_dir(Path(cfg.data_path) / Path(cfg.preprocessed_dir))
    create_dir(Path(cfg.data_path) / Path(cfg.feature_dir))
    create_dir(Path(cfg.data_path) / Path(cfg.ot_dir))
    create_dir(Path(cfg.data_path) / Path(cfg.results_dir))


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Setup directories
    RAW_PATH = Path(cfg.data_path) / Path(cfg.raw_dir)
    PREPROCESSED_PATH = Path(cfg.data_path) / Path(cfg.preprocessed_dir)
    FEATURE_PATH = Path(cfg.data_path) / Path(cfg.feature_dir)
    OT_PATH = Path(cfg.data_path) / Path(cfg.ot_dir)
    RESULTS_PATH = Path(cfg.data_path) / Path(cfg.results_dir)
    setup_directories(cfg)

    # Start timer
    start_time = time.time()

    # Get name of image configuration to be processed
    experiment_name = cfg.experiment_name

    ### PREPROCESSING ###
    # Check conf/preprocess.yaml for settings
    image_preprocessed_path = Path(PREPROCESSED_PATH / experiment_name)

    if not cfg.skip_preprocessing:
        # Create paths if they do not exist
        create_dir(image_preprocessed_path)
        preprocess_cuts_and_store_all(cfg, RAW_PATH, image_preprocessed_path, experiment_name)

    ### DROPLET DETECTION ###
    # Check conf/extract_droplets.yaml for settings
    image_feature_path = Path(FEATURE_PATH / experiment_name)

    if not cfg.skip_droplet_extraction:
        # Create paths if they do not exist
        create_dir(image_feature_path)
        detect_and_store_all(cfg, image_preprocessed_path, image_feature_path)

    ### DROPLET PATCHES EXTRACTION ###
    # Check conf/extract_droplets.yaml for settings

    if not cfg.skip_droplet_patch_extraction:
        # Create paths if they do not exist
        create_dir(image_feature_path)
        create_and_save_droplet_patches(cfg, image_preprocessed_path, image_feature_path)

    ### VISUAL EMBEDDING EXTRACTION ###
    # Check conf/extract_features.yaml for settings

    if not cfg.skip_visual_embedding_extraction:
        # Create paths if they do not exist
        create_dir(image_feature_path)
        create_and_save_droplet_embeddings(cfg, image_feature_path)

    ### TRACKING ###
    image_ot_path = Path(OT_PATH / experiment_name)
    if not cfg.skip_tracking:
        # Create paths if they do not exist
        create_dir(image_ot_path)

        test_features = np.random.rand(3, 2, 3)
        ot = OptimalTransport(cfg)
        ot.compute_and_store_ot_matrices_all(image_feature_path, image_ot_path)

    ### GENERATING RESULTS ###
    image_results_path = Path(RESULTS_PATH / experiment_name)

    if not cfg.skip_results_generation:
        # Create paths if they do not exist
        create_dir(image_results_path)
        compute_and_store_results_all(cfg, image_ot_path, image_results_path, image_feature_path)

    end_time = time.time()
    if cfg.verbose:
        print(f"Total processing time: {round(end_time - start_time)} seconds")


if __name__ == '__main__':
    main()
