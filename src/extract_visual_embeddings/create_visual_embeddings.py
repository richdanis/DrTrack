import os
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm
from omegaconf import DictConfig
import numpy as np

from extract_visual_embeddings.dataset import DropletDataset
from extract_visual_embeddings.models.efficientnet import EfficientNet


def load_model(cfg: DictConfig) -> torch.nn.Module:
    return EfficientNet(cfg)


def create_droplet_embeddings(cfg: DictConfig, dataset: DropletDataset, model: torch.nn.Module) -> Dict:
    """
    Create a dataframe with droplet embeddings.
    ----------
    Parameters:
    cfg: DictConfig:
        Global config.
    dataset: DropletDataset:
        A PyTorch dataset used to create a DataLoader.
    model: torch.nn.Module:
        Model that creates embeddings.
    ----------
    Returns:
    embeddings: Dict
        A dictionary with droplet_id, frame and visual embedding for each droplet.
    """
    print("model type", type(model))
    embeddings = {'embeddings': [], 'droplet_id': [], 'frame': []}

    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.extract_visual_embeddings.inference_batch_size,
                                         shuffle=False)

    for patch_batch, droplet_ids, frames in tqdm(loader, disable=not cfg.verbose):
        patch_batch = patch_batch.to(cfg.device)
        with torch.no_grad():

            embedding_batch = model(patch_batch)

            embedding_batch = embedding_batch.detach().cpu().numpy()

        for j in range(embedding_batch.shape[0]):
            embeddings['embeddings'].append(embedding_batch[j])

        embeddings['droplet_id'] += droplet_ids.numpy().tolist()
        embeddings['frame'] += frames.numpy().tolist()

    return embeddings


def create_and_save_droplet_embeddings(cfg: DictConfig, image_feature_path: Path):
    """
    Creates droplet patches of the preprocessed cuts and stores them in a npy file.
    ----------
    Parameters:
    cfg: DictConfig
       Global config.
    image_feature_path: Path:
        Directory where .csv files with droplet features are stored.
    """
    if cfg.verbose:
        print("\n===================================================================")
        print("Create droplet embeddings")
        print("===================================================================\n")
        print("Currently processing:")

    model = load_model(cfg)

    # Read droplet and cell tables from CSV files
    for filename in os.listdir(image_feature_path):
        f = os.path.join(image_feature_path, filename)
        # checking if it is a file
        if os.path.isfile(f) and filename.startswith("patches_"):
            cut_patch_file_name = filename

            if cfg.verbose:
                print(cut_patch_file_name)

            cut_patch_path = Path(image_feature_path / cut_patch_file_name)
            dataset = DropletDataset(cut_patch_path)

            droplet_embeddings_df = create_droplet_embeddings(cfg, dataset, model)

            droplet_base_file_name = cut_patch_file_name.replace("patches_", "")

            np.save(str(image_feature_path / f'embeddings_{droplet_base_file_name}'), droplet_embeddings_df)
