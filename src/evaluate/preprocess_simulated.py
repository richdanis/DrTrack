import jax
import jax.numpy as jnp
import pandas as pd
import torch
import numpy as np
from pathlib import Path
import random


class SimulatedData():
    """
    Class for loading and preprocessing simulated data
    """

    def __init__(self, cfg, image_simulated, image_processed_path, real_droplet_metadata_path=None):
        # Extract data
        self.image_processed_path = image_processed_path
        self.position_df = pd.read_csv(image_simulated, index_col=0)
        if real_droplet_metadata_path is None:
            self.random_num_cells = True
        else:
            self.random_num_cells = False
            # This file has index coming from the old detection, if we reindex them pandas should keep the order
            self.real_droplet_metadata_df = pd.read_csv(real_droplet_metadata_path, index_col=None)
            self.real_droplet_metadata_df.index = self.real_droplet_metadata_df.index.astype(str)
        self.experiment_name = cfg.experiment_name
        self.verbose = cfg.verbose
        self.args = cfg.evaluate

    def get_frame_position_jnp(self, frame, stride=2):
        x = jnp.array(self.position_df.iloc[::stride, frame])
        y = jnp.array(self.position_df.iloc[1::stride, frame])

        # Filter data if necessary
        if self.args.cutout_image:
            x = x[(x > self.args.cutout.x_min) & (x < self.args.cutout.x_max)]
            y = y[(y > self.args.cutout.y_min) & (y < self.args.cutout.y_max)]

        pos = jnp.array([x, y]).T

        return pos

    def get_frame_position_df(self, frame, real_data_frame, stride=2):
        x = jnp.array(self.position_df.iloc[::stride, frame])
        y = jnp.array(self.position_df.iloc[1::stride, frame])
        pos = jnp.array([x, y]).T
        index = self.position_df.index[::stride]
        index = [i[1:-2] for i in index]
        df = pd.DataFrame(pos, index=index, columns=["center_x", "center_y"])
        df["droplet_id"] = df.index
        df["radius"] = self.args.droplet_radius

        if self.random_num_cells:
            # Add number of cells randomly
            nr_cells_distribution = self.args.nr_cells_distribution
            max_num_cells = len(nr_cells_distribution)
            poss_num_cells = [i for i in range(max_num_cells)]
            num_droplets = len(df)

            df["nr_cells"] = random.choices(poss_num_cells, weights=nr_cells_distribution, k=num_droplets)
        else:
            print('ncb', self.real_droplet_metadata_df[f"nr_cells{real_data_frame}"])
            print('ind', index)
            df["nr_cells"] = self.real_droplet_metadata_df[f"nr_cells{real_data_frame}"]
            print('nc', df["nr_cells"])

        # Filter data if necessary
        if self.args.cutout_image:
            df = df[(df["center_x"] > self.args.cutout.x_min) & (df["center_x"] < self.args.cutout.x_max)]
            df = df[(df["center_y"] > self.args.cutout.y_min) & (df["center_y"] < self.args.cutout.y_max)]

        return df

    def get_plain_filtered_position_df(self, frame, stride=2):
        x = jnp.array(self.position_df.iloc[::stride, frame])
        y = jnp.array(self.position_df.iloc[1::stride, frame])
        pos = jnp.array([x, y]).T
        index = self.position_df.index[::stride]
        index = [i[:-2] for i in index]
        df = pd.DataFrame(pos, index=index, columns=["center_x", "center_y"])

        # Filter data if necessary
        if self.args.cutout_image:
            df = df[(df["center_x"] > self.args.cutout.x_min) & (df["center_x"] < self.args.cutout.x_max)]
            df = df[(df["center_y"] > self.args.cutout.y_min) & (df["center_y"] < self.args.cutout.y_max)]

        return df

    def create_and_store_position_dfs(self):
        # Progress
        if self.verbose:
            print("\n=========================================")
            print("Preprocessing Simulated Data")
            print("=========================================\n")
            print(f'Currently processing:')

        df_list = []
        for i, frame in enumerate(self.args.frames):
            # Progress
            if self.verbose:
                print(f'{self.experiment_name}: Frame {frame}')

            df = self.get_frame_position_df(frame=frame, real_data_frame=i, stride=self.args.stride * 2)
            df["frame"] = i
            df_list += [df]

        droplets_df = pd.concat(df_list)
        droplets_df.to_csv(Path(self.image_processed_path / f"droplets_.csv"), index=False)
