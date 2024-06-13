# Types
from omegaconf import DictConfig
from pathlib import Path

# Runtime measurement
import time

# Configuration library
import hydra

# Jax for fast computation using GPUs
import jax

# Local imports
from preprocess.for_all import preprocess_cuts_and_store_all
from detect_droplets.detect_and_store import detect_and_store_all
from extract_droplets.create_droplet_patches import create_and_save_droplet_patches
from extract_visual_embeddings.create_visual_embeddings import create_and_save_droplet_embeddings
from track.ot import OptimalTransport
from generate_results.get_trajectories import compute_and_store_results_all
from utils.file_structure import create_dir

def setup_directories(cfg):
    # Create directories if they do not exist
    create_dir(Path(cfg.data_path))
    create_dir(Path(cfg.data_path) / Path(cfg.raw_dir))
    create_dir(Path(cfg.data_path) / Path(cfg.preprocessed_dir))
    create_dir(Path(cfg.data_path) / Path(cfg.feature_dir))
    create_dir(Path(cfg.data_path) / Path(cfg.ot_dir))
    create_dir(Path(cfg.data_path) / Path(cfg.results_dir))


@hydra.main(config_path="../conf", config_name="config_track_droplets", version_base=None)
def main(cfg: DictConfig):
    """
    Main function for tracking droplets. The pipeline consists of the following steps:
    1. Preprocess the raw image and store as .npy arrays.
    2. Detect droplets and store as .csv files.
    3. Extract droplet patches and store as .npy arrays.
    4. Extract visual embeddings and store as .npy arrays.
    5. Compute optimal transport matrices and store as .npy arrays.
    6. Compute trajectories and store as .csv files.

    Note: Any step can be skipped by setting the corresponding flag in the configuration file.
    As long as the previous steps have been executed at some point (the corresponding files are present), the pipeline will work.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary.

    Returns
    -------
    None
    """

    ### SETUP ###
    # Set device
    if cfg.device == 'cpu':
        jax.devices("cpu")[0]

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
    # Check conf/preprocess for configurations.
    image_preprocessed_path = Path(PREPROCESSED_PATH / experiment_name)

    if not cfg.skip_preprocessing:
        # Create paths if they do not exist
        create_dir(image_preprocessed_path)
        preprocess_cuts_and_store_all(cfg, RAW_PATH, image_preprocessed_path)


    ### DROPLET DETECTION ###
    # Check conf/detect_droplets for configurations.
    # Path to store the detected droplets
    image_feature_path = Path(FEATURE_PATH / experiment_name)

    # Detect droplets and store as .csv files
    if not cfg.skip_droplet_detection:
        # Create paths if they do not exist
        create_dir(image_feature_path)
        detect_and_store_all(cfg, image_preprocessed_path, image_feature_path)


    ### DROPLET PATCHES EXTRACTION ###
    if not cfg.skip_droplet_patch_extraction:
        # Create paths if they do not exist
        create_dir(image_feature_path)
        create_and_save_droplet_patches(cfg, image_preprocessed_path, image_feature_path)


    ### VISUAL EMBEDDING EXTRACTION ###
    # Check conf/extract_features for configurations.

    if not cfg.skip_visual_embedding_extraction:
        # Create paths if they do not exist
        create_dir(image_feature_path)
        create_and_save_droplet_embeddings(cfg, image_feature_path)


    ### TRACKING ###
    # Check conf/track for configurations.
    image_ot_path = Path(OT_PATH / experiment_name)

    if not cfg.skip_tracking:
        # Create paths if they do not exist
        create_dir(image_ot_path)

        # Compute optimal transport matrices and store as .npy arrays
        ot = OptimalTransport(cfg)
        ot.compute_and_store_ot_matrices_all(image_feature_path, image_ot_path)


    ### GENERATING RESULTS ###
    # Check conf/generate_results for trajectory creation configurations.
    # Check conf/filter_results for trajectory filtering configurations.
    image_results_path = Path(RESULTS_PATH / experiment_name)

    if not cfg.skip_results_generation:
        # Create paths if they do not exist
        create_dir(image_results_path)

        # Compute trajectories and store full and filtered version as .csv files
        compute_and_store_results_all(cfg, image_ot_path, image_results_path, image_feature_path)

    # End timer
    end_time = time.time()
    if cfg.verbose:
        print(f"\nTotal processing time: {round(end_time - start_time)/60} minutes")

if __name__ == '__main__':
    main()
