import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class VisualizationDataset(Dataset):

    def __init__(self, dataset_path, metadata_path, skip_no_cells=True, both_channels=False):
        patches = np.load(dataset_path)
        metadata = pd.read_csv(metadata_path)
        assert patches.shape[0] == metadata.shape[0];
        "Incompatible shape of metadata and patches."
        self.dataset = []

        min_row = min(metadata['center_row0'])
        max_row = max(metadata['center_row0'])
        min_col = min(metadata['center_col0'])
        max_col = max(metadata['center_col0'])

        segment_num = 10
        segment_width = (max_col - min_col) // segment_num
        segment_height = (max_row - min_row) // segment_num
        segment_dict = {}

        for i in range(segment_num):
            for j in range(segment_num):
                segment_dict[(i * segment_width, (i + 1) * segment_width, i * segment_height,
                              (i + 1) * segment_height)] = segment_num * i + j

        for i in range(patches.shape[0]):
            for frame in range(patches.shape[1]):
                if skip_no_cells and metadata[f'nr_cells{frame}'][i] == 0:
                    continue
                patch = patches[i][frame]
                if not both_channels:
                    patch = patch[:1]
                patch = torch.from_numpy(patch)

                segment = segment_num * segment_num
                for key, val in segment_dict.items():
                    if (metadata[f'center_col{frame}'][i] >= key[0] and metadata[f'center_col{frame}'][i] < key[1] and
                            metadata[f'center_row{frame}'][i] >= key[2] and metadata[f'center_row{frame}'][i] < key[3]):
                        segment = val

                self.dataset.append((
                    patch,
                    {
                        'frame': frame,
                        'droplet_id': i,
                        'num_cells': metadata[f'nr_cells{frame}'][i],
                        'center_row': metadata[f'center_row{frame}'][i],
                        'center_col': metadata[f'center_col{frame}'][i],
                        'segment': segment
                    }
                ))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        return self.dataset[idx][0], self.dataset[idx][1]
