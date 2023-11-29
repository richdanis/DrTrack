import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch 

import matplotlib.pyplot as plt
from IPython import display

from ott import utils
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from pathlib import Path
import os

def create_trajectory_with_prob(cfg, ot_matrices):
    tracking_table = []

    for i, ot_matrix in enumerate(ot_matrices):

        threshold = 1/(ot_matrix.shape[0]*ot_matrix.shape[1])**2

        # masking entries that are below the threshold as True
        mask_threshold = ot_matrix < threshold

        rows_with_all_entries_less_threshold = np.all(mask_threshold, axis=1)
        cols_with_all_entries_less_threshold = np.all(mask_threshold, axis=0)
        
        # copy the ot matrix and set all values to zero that are below the threshold
        prob = ot_matrix.copy()
        prob = prob.at[mask_threshold].set(0.0)

        # set all columns to zero that have no connection (droplet is not in the current frame)
        prob = prob.at[:, cols_with_all_entries_less_threshold].set(0.0)

        # make row sum to 1 (to get probabilities)
        prob = prob/prob.sum(axis=1)[:, None]

        # mask rows that have no connection (droplet is not in the next frame)
        prob = prob.at[rows_with_all_entries_less_threshold, :].set(0.0)

        # only keep values that are positive
        ids_prob = prob > 0.0

        # combine the two masks
        ids_max_this = prob == prob.max(axis=1)[:, None]
        ids_max_next = prob == prob.max(axis=0)[None, :]
        
        ids_max = np.logical_and(ids_max_this, ids_max_next)
        ids = np.logical_and(ids_prob, ids_max)
        
        #######
        ids[np.where(ids_max_this.sum(axis=1) > 1), :] = False
        ids[:, np.where(ids_max_next.sum(axis=0) > 1)] = False
        #######

        this_frame_ids, next_frame_ids = np.where(ids)

        # if cfg.verbose:
        #     print(f"Frame {i} with {prob.shape[0]} droplets to frame {i+1} with {prob.shape[1]} droplets \n"
        #           f"matched droplets {len(this_frame_ids)} ({len(this_frame_ids)/np.min(prob.shape)*100:.2f}%)\n")
        
        # create a dataframe with the indices and the probabilities
        tmp = pd.DataFrame({f'frame': i,
                            f'droplet_id_this': this_frame_ids,
                            f'droplet_id_next': next_frame_ids,
                            f'prob': prob[this_frame_ids, next_frame_ids]})
        
        # append the dataframe to the tracking table
        tracking_table.append(tmp)

    tracking_table = pd.concat(tracking_table, axis=0)

    return tracking_table 

def process_and_merge_results(cfg, droplet_table: pd.DataFrame, 
                        tracking_table: pd.DataFrame) -> pd.DataFrame:

    max_droplets = droplet_table['droplet_id'].max() + 1
    full_prob = np.ones((max_droplets,), dtype=np.float32)

    result = pd.DataFrame({'droplet_id': np.arange(max_droplets)}, dtype=int)

    droplets = [df for _, df in droplet_table.groupby('frame')]

    result = result.merge(droplets[0], left_on='droplet_id', right_on='droplet_id', how='left').drop(columns='frame')
    result = result.rename(columns={'center_x': f'x0', 
                                    'center_y': f'y0', 
                                    'radius': f'r0',
                                    'nr_cells': f'nr_cells0'})
    result['droplet_id_next'] = pd.DataFrame({'droplet_id': np.arange(max_droplets)}, dtype=int)
    result = result.astype({'x0': 'Int32', 'y0': 'Int32','r0': 'Int32','nr_cells0': 'Int32'})
    for i, df in tracking_table.groupby('frame'):

        next = droplets[i+1]
        result = result.merge(df[['droplet_id_this', 'prob', 'droplet_id_next']], left_on='droplet_id_next', right_on='droplet_id_this', how='left')
        full_prob = full_prob * result['prob'].fillna(0.0).to_numpy()
        result = result.rename(columns={'prob': f'p{i}','droplet_id_next_y': 'droplet_id_next'}).drop(columns=['droplet_id_this','droplet_id_next_x',])
        result = result.merge(next, left_on='droplet_id_next', right_on='droplet_id', how='left').drop(columns='frame')
        result = result.rename(columns={'droplet_id_x': 'droplet_id',
                                        'center_x': f'x{i+1}', 
                                        'center_y': f'y{i+1}',
                                        'radius': f'r{i+1}',
                                        'nr_cells': f'nr_cells{i+1}'}).drop(columns='droplet_id_y')
        result = result.astype({f'x{i+1}': 'Int32', f'y{i+1}': 'Int32', f'r{i+1}': 'Int32',f'nr_cells{i+1}': 'Int32'})
        
    result['p'] = full_prob
    result = result.drop(columns=['droplet_id_next'])

    return result

def part_trajectory_prob(cfg, df):

    probs = []
    for col in df.columns:
        if col[0] == 'p' and col != 'p':
            probs.append(col)

    df = df[probs]

    # Calculate the product in sliding windows
    num_columns = df.shape[1]

    product_sliding_windows = [np.prod(df.iloc[:, i:j], axis=1) for i in range(num_columns) for j in range(i + 2, num_columns + 1)]

    # Create a new DataFrame with the results
    result_df = pd.DataFrame({
        f'p{i+1}-{j}': product
        for (i, j), product in zip(
            [(i, j) for i in range(num_columns) for j in range(i + 2, num_columns + 1)],
            product_sliding_windows
        )
    })

    # Display the result DataFrame
    return result_df

def compute_and_store_results_cut(cfg, cut_name, cut_ot_path, image_results_path, cut_feature_droplets_df):
    # load the ot matrices
    ot_matrices = []
    for file_name in os.listdir(cut_ot_path):
        if file_name.endswith(".npy"):
            ot_matrices.append(torch.load(cut_ot_path / file_name))
    
    trajectory_df = create_trajectory_with_prob(cfg, ot_matrices)
    results_df = process_and_merge_results(cfg, cut_feature_droplets_df, trajectory_df)
    part_probs = part_trajectory_prob(cfg, results_df)

    # concat the two dataframes
    final_results_df = pd.concat([results_df, part_probs], axis=1)

    # save the results
    final_results_df.to_csv(image_results_path / f'results_{cut_name}.csv', index=False)

def compute_and_store_results_all(cfg, image_ot_path, image_results_path, image_feature_path):
    # Progress
    if cfg.verbose:
        print("\n=========================================")
        print("Generating Results For all Cuts")
        print("=========================================\n")

        print(f'Currently generating results for cut:')
    
    # Iterate through all cuts
    for file_name in os.listdir(image_ot_path):
        if not file_name.startswith("ot_matrix_"):
            continue

        # Get cut name
        cut_name = file_name.replace("ot_matrix_", "")[:-4]

        # Progress
        if cfg.verbose:
            print(cut_name)

        # Get features of current cut
        cut_ot_path = Path(image_ot_path / file_name)
        cut_feature_droplets_file_path = Path(image_feature_path / f'droplets_{cut_name}.csv')
        cut_feature_droplets_df = pd.read_csv(cut_feature_droplets_file_path)

        # Compute and store ot matrices for current cut
        compute_and_store_results_cut(cfg, cut_name, cut_ot_path, image_results_path, cut_feature_droplets_df)