from pathlib import Path
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig


def structure_patches(cfg: DictConfig, gt_results_path: Path, gt_feature_path: Path, structured_patches_path: Path):
    """
    Create files with droplet patches and number of cells. Droplet keeps the same ID across frames.
    ----------
    Parameters:
    cfg: DictConfig:
        Global config.
    gt_results_path: Path:
        A path to the directory with trajectory files.
    gt_feature_path: Path:
        A path to the directory with patch files.
    structured_patches_path: Path:
        A path, where output patch and metadata files should be stored.
    ----------
    Returns:
    base_name: str
        A name (without extension) of the files with paired patches.
    """

    if cfg.verbose:
        print("=========================================")
        print("Extracting patches from GT data")
        print("=========================================")

    all_patches = {'droplet_id': [], 'frame': [], 'patch': []}
    all_num_cells = {f'nr_cells{f}': [] for f in range(cfg.gt_frames)}
    num_droplets = 0
    new_droplet_id = 0
    for dir_name in os.listdir(gt_results_path):
        if not dir_name.startswith("results_"):
            continue
        trajectory_path = gt_results_path / Path(dir_name)
        trajectories = pd.read_csv(trajectory_path)

        # Test for full trajectory uncertainty passing the threshold
        mask = trajectories[f'full_trajectory_uncertainty'] >= cfg.gt_uncertainty_threshold
        trajectories = trajectories[mask]

        # Test for the same number of cells
        mask = pd.Series(True, index=trajectories.index)
        for i in range(1, cfg.gt_frames):
            mask = mask & (trajectories[f'nr_cells{i}'] == trajectories['nr_cells0'])

        trajectories = trajectories[mask]

        # Test for max distance between frames
        mask = pd.Series(True, index=trajectories.index)
        for i in range(cfg.gt_frames - 1):
            mask = mask & ((trajectories[f'x{i}'] - trajectories[f'x{i + 1}']).abs() < cfg.gt_max_distance)
            mask = mask & ((trajectories[f'y{i}'] - trajectories[f'y{i + 1}']).abs() < cfg.gt_max_distance)

        trajectories = trajectories[mask]

        # Test for distance from the frame border
        min_x = 0
        max_x = trajectories['x0'].max()
        min_y = 0
        max_y = trajectories['y0'].max()
        mask = pd.Series(True, index=trajectories.index)
        for i in range(cfg.gt_frames):
            mask = mask & (trajectories[f'x{i}'] - min_x > cfg.gt_margin)
            mask = mask & (max_x - trajectories[f'x{i}'] > cfg.gt_margin)
            mask = mask & (trajectories[f'y{i}'] - min_y > cfg.gt_margin)
            mask = mask & (max_y - trajectories[f'y{i}'] > cfg.gt_margin)

        trajectories = trajectories[mask]

        patches_path = gt_feature_path / Path(dir_name.replace('results', 'patches').replace('csv', 'npy'))
        patches = pd.DataFrame(np.load(patches_path, allow_pickle=True).item())

        # We want the matched droplets to share the droplet ID across frames
        for trajectory_id in trajectories.index:
            for frame in range(cfg.gt_frames):
                all_patches['droplet_id'].append(new_droplet_id)
                all_patches['frame'].append(frame)
                patch_row = patches[
                    (patches['frame'] == frame) & (patches['droplet_id'] == trajectories[f'id_{frame}'][trajectory_id])]
                all_patches['patch'].append(patch_row['patch'].item())

                all_num_cells[f'nr_cells{frame}'].append(trajectories[f'nr_cells{frame}'][trajectory_id])
            new_droplet_id += 1
        num_droplets += len(trajectories)

    all_num_cells_df = pd.DataFrame(all_num_cells)
    base_name = f'paired_patches_{cfg.gt_frames}_frames_{num_droplets}_droplets'
    np.save(str(structured_patches_path / Path(f'{base_name}.npy')), all_patches, allow_pickle=True)
    all_num_cells_df.to_csv(structured_patches_path / Path(f'{base_name}_metadata.csv'))
    return base_name


# Used with paired_path = "c334_unstimulated_crop4x4.npy"
def generate_paired_patches(paired_path, data_path):
    patches = np.load("c334_unstimulated_crop4x4.npy")

    num_droplets = patches.shape[0]
    num_frames = patches.shape[1]

    droplet_patches = {'droplet_id': [], 'frame': [], 'patch': []}

    for i in range(num_frames):
        for j in range(num_droplets):
            droplet_patches["droplet_id"].append(j)
            droplet_patches["frame"].append(i)
            droplet_patches["patch"].append(patches[j, i, :, :, :])

    np.save("paired_patches.npy", droplet_patches, allow_pickle=True)

    return droplet_patches


if __name__ == '__main__':
    # generate_paired_patches('', '')
    # Visual sanity check for matching patches
    # patches = pd.DataFrame(
    #     np.load("../../evaluation/03_features/paired_patches_9_frames_3277_droplets.npy", allow_pickle=True).item())
    patches = pd.DataFrame(np.load(
        "/home/weronika/Documents/masters/sem3/data_science_lab/data/03_features/small_mvt_3_full/patches_small_mvt_3_full_y0_x6278.npy",
        allow_pickle=True).item())
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    patch = patches[patches['droplet_id'] == 950].reset_index()
    for i, row in patch.iterrows():
        image = row['patch']
        row_position, col_position = divmod(i, 3)
        axs[row_position, col_position].imshow(image[0, :, :])
        axs[row_position, col_position].axis('off')  # Optional: Turn off axis labels

    plt.tight_layout()
    plt.show()
