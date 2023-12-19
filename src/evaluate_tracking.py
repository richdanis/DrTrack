import os
from pathlib import Path
import numpy as np
import shutil
import jax
import pandas as pd
import wandb
import time
import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

from extract_visual_embeddings.create_visual_embeddings import create_and_save_droplet_embeddings
from track.ot import OptimalTransport
from generate_results.get_trajectories import compute_and_store_results_all
from evaluate.preprocess_simulated import SimulatedData
from evaluate.get_scores import OtEvaluation
from evaluate.calibration_plot import save_calibration_plot
from evaluate.generate_paired_patches import structure_patches


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


@hydra.main(config_path="../conf", config_name="config_evaluate_tracking", version_base=None)
def main(cfg: DictConfig):
    if cfg.device == 'cpu':
        jax.devices("cpu")[0]

    if cfg.wandb:
        os.environ["WANDB__SERVICE_WAIT"] = "300"

        wandb.init(
            project="DrTrack",
            entity="dslab-23",
            config=OmegaConf.to_container(cfg),
            dir="logs"
        )
    # Setup directories
    SIMULATED_PATH = Path(cfg.data_path) / Path(cfg.simulated_dir)
    PREPROCESSED_PATH = Path(cfg.data_path) / Path(cfg.preprocessed_dir)
    FEATURE_PATH = Path(cfg.data_path) / Path(cfg.feature_dir)
    OT_PATH = Path(cfg.data_path) / Path(cfg.ot_dir)
    RESULTS_PATH = Path(cfg.data_path) / Path(cfg.results_dir)

    ### EXTRACT PATCHES ###
    if cfg.extract_patches:
        GT_FEATURE_PATH = Path(cfg.gt_data_path) / Path(cfg.gt_feature_dir) / Path(cfg.gt_experiment_name)
        GT_RESULTS_PATH = Path(cfg.gt_data_path) / Path(cfg.gt_results_dir) / Path(cfg.gt_experiment_name)
        base_name = structure_patches(cfg, GT_RESULTS_PATH, GT_FEATURE_PATH, FEATURE_PATH)
        paired_patches = f'{base_name}.npy'
        paired_patches_metadata = f'{base_name}_metadata.csv'

        if cfg.verbose:
            print("***")
            print(f'Extracted patches from GT data and saved them as: {paired_patches}')
            print("***")
    else:
        paired_patches = cfg.paired_patches
        paired_patches_metadata = cfg.paired_patches_metadata

        if cfg.verbose:
            print("***")
            print(f'Using paired patches from: {paired_patches}')
            print("***")

    ### PREPROCESSING ###
    # Preprocess simulated data
    image_simulated = Path(SIMULATED_PATH / cfg.simulated_image)
    image_preprocessed_path = Path(PREPROCESSED_PATH / cfg.experiment_name)
    image_feature_path = Path(FEATURE_PATH / cfg.experiment_name)

    # for sweep
    if cfg.sweep:
        if cfg.extract_visual_embeddings.name == "droplets_all":
            if cfg.simulated_image == "small_mvt_6000_droplets.csv":
                image_feature_path = Path(FEATURE_PATH / "small_all")
            elif cfg.simulated_image == "medium_mvt_6000_droplets.csv":
                image_feature_path = Path(FEATURE_PATH / "medium_all")
            elif cfg.simulated_image == "large_mvt_6000_droplets.csv":
                image_feature_path = Path(FEATURE_PATH / "large_all")
        elif cfg.extract_visual_embeddings.name == "droplets_only_cells":
            if cfg.simulated_image == "small_mvt_6000_droplets.csv":
                image_feature_path = Path(FEATURE_PATH / "small_only_cells")
            elif cfg.simulated_image == "medium_mvt_6000_droplets.csv":
                image_feature_path = Path(FEATURE_PATH / "medium_only_cells")
            elif cfg.simulated_image == "large_mvt_6000_droplets.csv":
                image_feature_path = Path(FEATURE_PATH / "large_only_cells")

    create_dir(image_preprocessed_path)
    create_dir(image_feature_path)

    if not cfg.skip_preprocessing:
        if len(paired_patches_metadata) == 0:
            real_droplet_metadata_path = None
        else:
            real_droplet_metadata_path = Path(FEATURE_PATH / paired_patches_metadata)
        sim_data = SimulatedData(cfg, image_simulated, image_feature_path, real_droplet_metadata_path)
        sim_data.create_and_store_position_dfs()

    ### VISUAL EMBEDDING EXTRACTION ###
    # Check conf/extract_features.yaml for settings

    if not cfg.skip_visual_embedding_extraction:
        # Create paths if they do not exist
        create_dir(image_feature_path)

        # Copy paired patches to data_path/feature_dir
        paired_patches_path = FEATURE_PATH / Path(paired_patches)
        new_name = "patches_.npy"
        new_path = image_feature_path / Path(new_name)

        # Copy to required location if it does not exist
        if not os.path.exists(new_path):
            shutil.copyfile(paired_patches_path, new_path)

        # Create embeddings using configured model
        create_and_save_droplet_embeddings(cfg, image_feature_path)

    ### TRACKING ###
    # change experiment name to timestamp
    ## The following should only be used for parameter sweeps - otherwise, when running the script manually, the experiment name should be set in the config file
    if cfg.sweep:
        cfg.experiment_name = "sweep"
    image_ot_path = Path(OT_PATH / cfg.experiment_name)

    if not cfg.skip_tracking:
        # Create paths if they do not exist
        create_dir(image_ot_path)

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

        if not cfg.skip_calibration_plot:
            save_calibration_plot(cfg, image_results_path)

    if cfg.wandb:
        wandb.log({"alpha": cfg.track.alpha,
                   "dist": cfg.track.embedding_dist,
                   "tau": cfg.track.tau_a,
                   "relative_epsilon": cfg.track.relative_epsilon}, step=7)
        wandb.finish()


if __name__ == '__main__':
    main()
