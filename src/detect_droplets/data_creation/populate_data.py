import os
import re
from pathlib import Path
import toml
import pandas as pd
import random
import logging
import torch
import numpy as np
from models.models_common import create_embeddings, FolderDataset
from models.ViTMAE import ViTMAE
from . import droplet_retriever, droplets_and_cells


# def raw_to_preprocessed_alt_franc(raw_image_path: Path, 
#                                   image_name: str, 
#                                   PREPROCESSED_PATH: Path,
#                                   pixel:int = -1) -> np.ndarray:
#     preprocessed_image = preprocessing.preprocess_alt_franc(raw_image_path, pixel)
#     path = Path(PREPROCESSED_PATH / f"preprocessed_drpdtc_{image_name}.npy")
#     np.save(path, preprocessed_image)
#     return preprocessed_image


# def raw_to_preprocessed_featextr(raw_image_path: Path, image_name: str, PREPROCESSED_PATH: Path, pixel:int = -1) -> np.ndarray:
#     preprocessed_image = preprocessing.preprocess_alt_featextr(raw_image_path, pixel=pixel)
#     path = Path(PREPROCESSED_PATH / f"preprocessed_featextr_bf_{image_name}.npy")
#     np.save(path, preprocessed_image[:, 0, :, :])
#     path = Path(PREPROCESSED_PATH / f"preprocessed_featextr_dapi_{image_name}.npy")
#     np.save(path, preprocessed_image[:, 1, :, :])
#     return preprocessed_image


def save_droplet_images(dataset: np.ndarray, image_name: str, DROPLET_PATH: Path) -> None:

    folder_path = Path(DROPLET_PATH / image_name)

    try:
        os.mkdir(folder_path)
    except FileExistsError as _:
        pass

    for se in dataset:
        frame, id, patch = se
        patch = np.float64(droplet_retriever.resize_patch(patch, 64))
        patch[patch == 0.0] = np.nan
        thresh = np.nanquantile(patch, 0.5, axis = (1, 2))
        patch[np.isnan(patch)] = 0.0
        patch = np.uint16(np.clip((patch - thresh[:, None, None]) / (2**16 - 1 - thresh[:, None, None]), 0, 1.0) * (2**16 - 1))
        patch[np.isnan(patch)] = 0.0
        np.save(Path(folder_path / ("f" + str(frame) + "_d" + str(id).zfill(4))), patch)


def prep_experiment_dir(EXPERIMENTS_PATH, experiment):
    experiment_base_dir = os.path.join(EXPERIMENTS_PATH, str(experiment))
    try:
        os.mkdir(experiment_base_dir)
    except FileExistsError as _:
        pass
    try:
        experiment_idx = max([int(max(n.lstrip("0"), "0")) for n in os.listdir(experiment_base_dir)])
    except ValueError as _:
        experiment_idx = 0
    experiment_dir = os.path.join(experiment_base_dir, str(experiment_idx).zfill(3))
    try:
        os.mkdir(experiment_dir)
    except FileExistsError as _:
        pass
    return Path(experiment_dir)


def print_cuda_status():
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')


def populate(raw_image_path, image_name, FEATURE_PATH, PREPROCESSED_PATH, DROPLET_PATH, EXPERIMENT_PATH = None,
             omit_droplet_dataset = False,
             create_emb = False,
             train_model = False,
             radius_min = 12,
             radius_max = 25,
             pixel = -1) -> None:

    print("Image preprocessing for droplet detection ...")
    preprocessed_image = raw_to_preprocessed_alt_franc(raw_image_path, image_name, PREPROCESSED_PATH, pixel)

    print("Detecting droplets and cells...")
    droplet_feature_path = Path(FEATURE_PATH / f"droplets_{image_name}.csv")
    cell_feature_path = Path(FEATURE_PATH / f"cells_{image_name}.csv")
    droplets_and_cells.generate_output_from_ndarray(preprocessed_image, 
                                                    droplet_feature_path, 
                                                    cell_feature_path, 
                                                    True, "", False, 
                                                    radius_min = radius_min, 
                                                    radius_max = radius_max)

    print("Image preprocessing for feature extraction ...")
    preprocessed_image = raw_to_preprocessed_featextr(raw_image_path, image_name, PREPROCESSED_PATH, pixel)
        
    if not omit_droplet_dataset:
        print("Creating droplet images...")
        df_full_droplet, droplet_patches = droplet_retriever.create_dataset_cell_enhanced_from_ndarray(
                                                                           preprocessed_image, 
                                                                           droplet_feature_path, 
                                                                           cell_feature_path, 
                                                                           buffer = -2, 
                                                                           suppress_rest = True, 
                                                                           suppression_slack = -3)
        save_droplet_images(droplet_patches, image_name, DROPLET_PATH)
        path = Path(FEATURE_PATH/ f"fulldataset_{image_name}.csv")
        df_full_droplet.to_csv(path, index=False)
        
    else:
        print("Omitting droplet dataset creation...")
    
    model = None
    config = None
    if train_model:
        print_cuda_status()
        experiment_dir = prep_experiment_dir(EXPERIMENT_PATH, image_name)
        config = toml.load(EXPERIMENT_PATH/"model.toml")["config"]

        torch.manual_seed(config["seed"])
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        if config["train_dataset"] == '':
            config["train_dataset"] = Path(DROPLET_PATH, f"{image_name}")
            config["val_dataset"] = Path(DROPLET_PATH, f"{image_name}")
        config['experiment_dir'] = experiment_dir

        logging.basicConfig(filename=str(experiment_dir / 'output.log'), level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler())

        if "checkpoint" in config:
            model = ViTMAE(config)
            checkpoint_path = config["checkpoint"]
            checkpoint = torch.load(checkpoint_path, map_location=model.device)
            model.model.load_state_dict(checkpoint['model_state_dict'])
            model.step = checkpoint['step']
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("loaded checkpoint", checkpoint_path)
        else:
            model = ViTMAE(config)

        logging.info("Training model...")
        model.train()

    if create_emb:
        print("Creating Embeddings")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'


        if not train_model:
            config = toml.load(EXPERIMENT_PATH / "0003.toml")["config"]

            checkpoint = torch.load(config["checkpoint"], map_location=device)

            model = ViTMAE(config)
            model.model.load_state_dict(checkpoint['model_state_dict'])

        transforms = None
        train_dataset = FolderDataset(Path(DROPLET_PATH, f"{image_name}"), transforms)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

        embeddings = create_embeddings(device, train_dataloader, config, model.model)
        embeddings = embeddings.reshape((embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2]))
        indexes = np.array([0,0])
        marker1, marker2, marker3 = 'f', '_d', '.npy'
        regex1 = marker1 + '(.+?)' + marker2
        regex2 = marker2 + '(.+?)' + marker3
        for i in sorted(os.listdir(Path(DROPLET_PATH, f"{image_name}"))):
            frame = int(re.search(regex1, i).group(1))
            droplet_id = int(re.search(regex2, i).group(1))
            indexes = np.vstack((indexes, np.array([frame, droplet_id])))

        indexes = indexes[1:, :]
        embeddings = np.concatenate((indexes, embeddings), axis=1)
        pd.DataFrame(embeddings).to_csv(Path(FEATURE_PATH / f"embeddings_{image_name}.csv"), header = False, index = False)

