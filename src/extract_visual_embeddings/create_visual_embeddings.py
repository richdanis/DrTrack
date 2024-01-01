import os

# Types
from pathlib import Path
from typing import Dict
from omegaconf import DictConfig

# Handling neural networks
import torch
import numpy as np

# Progress
from tqdm import tqdm

# Local imports
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

    embeddings = {'embeddings': [], 'droplet_id': [], 'frame': []}

    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.extract_visual_embeddings.inference_batch_size,
                                         shuffle=False)

    model = model.to(cfg.device)
    model.eval()
    for patch_batch, droplet_ids, frames, cell_nrs in tqdm(loader, disable=not cfg.verbose):
        patch_batch = patch_batch.to(cfg.device)

        # Embed only droplets with cells or all droplets depending on the flag
        if cfg.extract_visual_embeddings.embed_without_cells:
            embedding_mask = torch.ones_like(cell_nrs, dtype=torch.bool)
        else:
            embedding_mask = cell_nrs > 0
        # Necessary for batches of size 1, as torch tensor of length 1 will be interpreted as a scalar.
        embedding_mask = embedding_mask.tolist()

        embedding_batch = np.zeros((patch_batch.shape[0], cfg.extract_visual_embeddings.embed_dim))
        if np.sum(embedding_mask) > 0:
            with torch.no_grad():
                embedding_batch[embedding_mask] = model(patch_batch[embedding_mask]).detach().cpu().numpy()
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
        print("Create Droplet Embeddings")
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

            droplet_base_file_name = cut_patch_file_name.replace("patches_", "").replace(".npy", "")
            cut_droplet_path = Path(image_feature_path / f"droplets_{droplet_base_file_name}.csv")

            dataset = DropletDataset(cfg, cut_patch_path, cut_droplet_path)

            droplet_embeddings_df = create_droplet_embeddings(cfg, dataset, model)

            np.save(str(image_feature_path / f'embeddings_{droplet_base_file_name}.npy'), droplet_embeddings_df)
