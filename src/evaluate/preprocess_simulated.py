# Handling paths
from pathlib import Path
from typing import Optional, Tuple, List
from omegaconf import DictConfig

# Jax for fast computation using GPUs
import jax.numpy as jnp

# Data handling
import pandas as pd

# Other imports
import random


class SimulatedData():
    """
    Class for loading and preprocessing simulated data.

    Parameters
    ----------
    cfg : object
        Configuration object containing experiment settings.
    image_simulated : str
        Path to the simulated image file.
    image_processed_path : str
        Path to store the processed image file.
    real_droplet_metadata_path : str, optional
        Path to the real droplet metadata file. Defaults to None.

    Attributes
    ----------
    image_processed_path : str
        Path to the processed image file.
    position_df : pandas.DataFrame
        DataFrame containing the position data.
    random_num_cells : bool
        Flag indicating whether to use random number of cells.
    real_droplet_metadata_df : pandas.DataFrame
        DataFrame containing the real droplet metadata.
    experiment_name : str
        Name of the experiment.
    verbose : bool
        Flag indicating whether to print progress messages.
    args : object
        Evaluation arguments.

    Methods
    -------
    get_frame_position_jnp(frame, stride=2)
        Get the position of droplets in a specific frame using JAX NumPy.
    get_frame_position_df(frame, real_data_frame, stride=2)
        Get the position of droplets in a specific frame as a DataFrame.
    get_plain_filtered_position_df(frame, stride=2)
        Get the position of droplets in a specific frame as a filtered DataFrame.
    create_and_store_position_dfs()
        Create and store position DataFrames for all frames.
    """
    def __init__(self, 
                 cfg: DictConfig, 
                 image_simulated: Path, 
                 image_processed_path: Path, 
                 real_droplet_metadata_path: Path = None) -> None:
        """
        Initialize the PreprocessSimulated class with parameters described in the class docstring.
        """
        # Extract data
        self.image_processed_path = image_processed_path
        self.position_df = pd.read_csv(image_simulated, index_col=0)

        # Check if we have real data. If not, we will use a random number of cells per droplet.
        if real_droplet_metadata_path is None:
            self.random_num_cells = True
        else:
            self.random_num_cells = False
            # This file has index coming from the old detection, if we reindex them pandas should keep the order
            self.real_droplet_metadata_df = pd.read_csv(real_droplet_metadata_path, index_col=None)
            self.real_droplet_metadata_df.index = self.real_droplet_metadata_df.index.astype(str)

        # Other attributes
        self.experiment_name = cfg.experiment_name
        self.verbose = cfg.verbose
        self.args = cfg.evaluate

    def get_frame_position_jnp(self, 
                               frame: int, 
                               stride: int = 2) -> jnp.array:
        """
        Get the position of droplets in a specific frame as JAX NumPy array.

        Parameters
        ----------
        frame : int
            Frame number.
        stride : int, optional
            Stride value for subsampling the position data. Defaults to 2.

        Returns
        -------
        pos : jnp.array
            Array containing the positions of droplets in the frame.
        """
        # Get position data
        x = jnp.array(self.position_df.iloc[::stride, frame])
        y = jnp.array(self.position_df.iloc[1::stride, frame])

        # Filter data if necessary
        if self.args.cutout_image:
            x = x[(x > self.args.cutout.x_min) & (x < self.args.cutout.x_max)]
            y = y[(y > self.args.cutout.y_min) & (y < self.args.cutout.y_max)]

        # Create position array
        pos = jnp.array([x, y]).T

        return pos

    def get_frame_position_df(self, 
                              frame: int, 
                              real_data_frame: int, 
                              stride: int = 2) -> pd.DataFrame:
        """
        Get the position of droplets in a specific frame as a DataFrame.

        Parameters
        ----------
        frame : int
            Frame number.
        real_data_frame : int
            Real data frame number.
        stride : int, optional
            Stride value for subsampling the position data. Defaults to 2.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame containing the positions of droplets in the frame.
        """
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
            # print('ncb', self.real_droplet_metadata_df[f"nr_cells{real_data_frame}"])
            # print('ind', index)
            df["nr_cells"] = self.real_droplet_metadata_df[f"nr_cells{real_data_frame}"]
            # print('nc', df["nr_cells"])

        # Filter data if necessary
        if self.args.cutout_image:
            df = df[(df["center_x"] > self.args.cutout.x_min) & (df["center_x"] < self.args.cutout.x_max)]
            df = df[(df["center_y"] > self.args.cutout.y_min) & (df["center_y"] < self.args.cutout.y_max)]

        return df

    def get_plain_filtered_position_df(self, 
                                       frame: int, 
                                       stride: int = 2) -> pd.DataFrame:
        """
        Get the position of droplets in a specific frame as a filtered DataFrame.

        Parameters
        ----------
        frame : int
            Frame number.
        stride : int, optional
            Stride value for subsampling the position data. Defaults to 2.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame containing the positions of droplets in the frame.
        """
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

    def create_and_store_position_dfs(self) -> None:
        """
        Create and store position DataFrames for all frames.
        
        Returns
        -------
        None
        """
        # Progress
        if self.verbose:
            print("\n=========================================")
            print("Preprocessing Simulated Data")
            print("=========================================\n")
            print(f'Currently processing: \n{self.experiment_name}')

        df_list = []
        for i, frame in enumerate(self.args.frames):
            # Progress
            if self.verbose:
                print(f'- Frame {frame}')

            df = self.get_frame_position_df(frame=frame, real_data_frame=i, stride=self.args.stride * 2)
            df["frame"] = i
            df_list += [df]

        droplets_df = pd.concat(df_list)
        droplets_df.to_csv(Path(self.image_processed_path / f"droplets_.csv"), index=False)
