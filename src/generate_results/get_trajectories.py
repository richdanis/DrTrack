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

def get_epsilon_dependent_id_mapping(cfg, ot_matrix):
    """
    This function returns a mapping from the droplet ids in the current frame to the droplet ids in the next frame 
    based on the entries of the OT matrix. The mapping is epsilon dependent.
    """
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

    probs = prob[this_frame_ids, next_frame_ids]

    return this_frame_ids, next_frame_ids, probs

def scale_ot_matrix(cfg, ot_matrix):
    """
    This function scales the OT matrix to a range of [0,1].
    """
    # Scale onto 0-1 range
    ot_matrix_scaled = (ot_matrix - ot_matrix.min()) / (ot_matrix.max() - ot_matrix.min())

    return ot_matrix_scaled

def transform_to_rank_based_probability_matrix(cfg, ot_matrix):
    """
    This function transforms the OT matrix to a probability matrix based on the ranks of the entries in the OT matrix.
    """
    # Flatten the matrix into 1D array
    flattened = ot_matrix.flatten()

    # Get the ordering of the elements
    ranks = flattened.argsort().argsort()

    # Reshape the ordering to match the original matrix shape
    ranks = ranks.reshape(ot_matrix.shape)

    # Make sure the the maximal probability is 1
    min_dim = min(ot_matrix.shape[0], ot_matrix.shape[1])
    ranks = ranks // min_dim

    # Add row and col based ordering to deal with ties
    row_ranks = ot_matrix.argsort().argsort()
    col_ranks = ot_matrix.argsort(axis=0).argsort(axis=0)
    ranks_row_col = row_ranks + col_ranks
    norm_ranks_row_col = ranks_row_col / (ot_matrix.shape[0] + ot_matrix.shape[1] - 2)

    # Transform to probability matrix
    total_ranks = ranks + norm_ranks_row_col

    # Apply temperature for more constrast in the probabilities
    temp = 2
    prob_matrix = total_ranks ** temp / (min_dim+1) ** temp

    return prob_matrix

def get_indices_to_drop(cfg, probs, vec):
    """
    This function returns the indices of the entries that should be dropped.
    Indices are dropped if they have a duplicate entry in vec and the probability is lower than the maximum probability.
    """
    # Find duplicate entries in next_frame_ids
    unique, counts = np.unique(vec, return_counts=True)
    duplicates = unique[counts > 1]

    # Find the indices of the duplicates
    idxs_to_remove = []
    probs_np = np.array(probs)
    for duplicate in duplicates:
        # Find the indices of the duplicates
        indices = np.where(vec == duplicate)[0]

        # Find the index of the maximum probability
        max_index = indices[probs_np[indices].argmax()]

        # Remove the all duplicates with the exception of the maximum probability
        idxs_to_remove.extend(indices[indices != max_index])

    return idxs_to_remove


def get_epsilon_independent_id_mapping(cfg, ot_matrix, frame_id):
    """
    This function returns a mapping from the droplet ids in the current frame to the droplet ids in the next frame.
    The mapping is epsilon independent and based on ranks.
    """
    # Get probability matrix based on ranks
    if cfg.generate_results.uncertainty_type == "prob_matrix":
        prob_matrix = transform_to_rank_based_probability_matrix(cfg, ot_matrix)

        # Save probability matrices for scoring
        name = f"{frame_id}-{frame_id+1}.npy"
        directory_name = cfg.generate_results.uncertainty_type + "_" + cfg.experiment_name
        directory_path = Path(cfg.data_path) / Path(cfg.ot_dir) / Path(cfg.experiment_name) / Path(directory_name)

        torch.save(prob_matrix, directory_path / name)

    elif cfg.generate_results.uncertainty_type == "scaled_ot":
        prob_matrix = scale_ot_matrix(cfg, ot_matrix)

        # Save scaled ot matrix for scoring
        # Create directory if it does not exist
        name = f"{frame_id}-{frame_id+1}.npy"
        directory_name = cfg.generate_results.uncertainty_type + "_" + cfg.experiment_name
        directory_path = Path(cfg.data_path) / Path(cfg.ot_dir) / Path(cfg.experiment_name) / Path(directory_name)

        # Min-max scaling
        ot_matrix_scaled = (ot_matrix - ot_matrix.min()) / (ot_matrix.max() - ot_matrix.min())
        torch.save(ot_matrix_scaled, directory_path / name)

    elif cfg.prob_matrix_type == "original":
        return get_epsilon_dependent_id_mapping(cfg, ot_matrix)


    # Get the axis with maximal dimension
    max_axis = np.argmax(prob_matrix.shape)

    # Get the indices of the maximal values and the corresponding probabilities
    # Get argmax in dimension of max_axis
    max_axis = np.argmax(ot_matrix.shape)

    if max_axis == 0:
        next_frame_ids = np.argmax(ot_matrix, axis=1)
        this_frame_ids = np.arange(ot_matrix.shape[0])
        probs = [prob_matrix[i, next_frame_ids[i]] for i in this_frame_ids]

        # Drop all entries with duplicate next id and lowest probability
        idx_to_remove = get_indices_to_drop(cfg, probs, next_frame_ids)

        # Remove the duplicates
        this_frame_ids = np.delete(this_frame_ids, idx_to_remove)
        next_frame_ids = np.delete(next_frame_ids, idx_to_remove)
        probs = np.delete(probs, idx_to_remove)

    elif max_axis == 1:
        next_frame_ids = np.arange(ot_matrix.shape[1])
        this_frame_ids = np.argmax(ot_matrix, axis=0)
        probs = [prob_matrix[this_frame_ids[i], i] for i in next_frame_ids]

        # Drop all entries with duplicate target ids
        idx_to_remove = get_indices_to_drop(cfg, probs, this_frame_ids)

        # Remove the duplicates
        this_frame_ids = np.delete(this_frame_ids, idx_to_remove)
        next_frame_ids = np.delete(next_frame_ids, idx_to_remove)
        probs = np.delete(probs, idx_to_remove)

    else :
        raise ValueError("Something went wrong with OT matrix shape.")

    # Make sure to include a possible trajectory for all droplets
    # if max_axis == 0:
    #     missing_ids = np.setdiff1d(np.arange(ot_matrix.shape[1]), next_frame_ids)
    #     probs_missing = [prob_matrix[this_frame_ids[i], i] for i in missing_ids]

    # elif max_axis == 1:
    #     missing_ids = np.setdiff1d(np.arange(ot_matrix.shape[0]), this_frame_ids)
    #     probs_missing = [prob_matrix[i, next_frame_ids[i]] for i in missing_ids]

    # # Add ids and probs to the rest
    # this_frame_ids = np.concatenate((this_frame_ids, missing_ids))
    # next_frame_ids = np.concatenate((next_frame_ids, missing_ids))
    # probs = np.concatenate((probs, probs_missing))

    return this_frame_ids, next_frame_ids, probs

def create_trajectory_with_prob(cfg, ot_matrices):
    tracking_table = []

    for i, ot_matrix in enumerate(ot_matrices):
        # get the mapping from the current frame to the next frame
        this_frame_ids, next_frame_ids, probs = get_epsilon_independent_id_mapping(cfg, ot_matrix, i)

        # if cfg.verbose:
        #     print(f"Frame {i} with {prob.shape[0]} droplets to frame {i+1} with {prob.shape[1]} droplets \n"
        #           f"matched droplets {len(this_frame_ids)} ({len(this_frame_ids)/np.min(prob.shape)*100:.2f}%)\n")
        
        # create a dataframe with the indices and the probabilities
        tmp = pd.DataFrame({f'frame': i,
                            f'droplet_id_this': this_frame_ids,
                            f'droplet_id_next': next_frame_ids,
                            f'prob': probs})
        
        # append the dataframe to the tracking table
        tracking_table.append(tmp)

    tracking_table = pd.concat(tracking_table, axis=0)

    return tracking_table 

def process_and_merge_results(cfg, droplet_table: pd.DataFrame, 
                        tracking_table: pd.DataFrame) -> pd.DataFrame:

    max_droplets = droplet_table['droplet_id'].max() + 1
    full_prob = np.ones((max_droplets,), dtype=np.float64)

    result = pd.DataFrame({'droplet_id': np.arange(max_droplets)}, dtype=int)

    droplets = [df for _, df in droplet_table.groupby('frame')]

    result = result.merge(droplets[0], left_on='droplet_id', right_on='droplet_id', how='left').drop(columns='frame')
    result = result.rename(columns={'center_x': f'x0', 
                                    'center_y': f'y0', 
                                    'radius': f'r0',
                                    'nr_cells': f'nr_cells0'})
    result['droplet_id_next'] = pd.DataFrame({'droplet_id': np.arange(max_droplets)}, dtype=int)
    result = result.astype({'x0': float, 'y0': float,'r0': 'Int32','nr_cells0': 'Int32'})
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
        result = result.astype({f'x{i+1}': float, f'y{i+1}': float, f'r{i+1}': 'Int32',f'nr_cells{i+1}': 'Int32'})
        
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
    for file_name in sorted(os.listdir(cut_ot_path), key=lambda x: int(x.split("-")[0])):
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
    
    # Create directory if it does not exist
    directory_name = "prob_matrix_" + cfg.experiment_name
    directory_path = Path(cfg.data_path) / Path(cfg.ot_dir) / Path(cfg.experiment_name) / Path(directory_name)

    # Create directory if it does not exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Create directory if it does not exist
    directory_name = "scaled_ot_" + cfg.experiment_name
    directory_path = Path(cfg.data_path) / Path(cfg.ot_dir) / Path(cfg.experiment_name) / Path(directory_name)

    # Create directory if it does not exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Iterate through all cuts
    for dir_name in os.listdir(image_ot_path):
        if not dir_name.startswith("ot_matrix_"):
            continue

        # Get cut name
        cut_name = dir_name.replace("ot_matrix_", "")

        # Progress
        if cfg.verbose:
            print(cut_name)

        # Get features of current cut
        cut_ot_path = Path(image_ot_path / dir_name)
        cut_feature_droplets_file_path = Path(image_feature_path / f'droplets_{cut_name}.csv')
        cut_feature_droplets_df = pd.read_csv(cut_feature_droplets_file_path)

        # Compute and store ot matrices for current cut
        compute_and_store_results_cut(cfg, cut_name, cut_ot_path, image_results_path, cut_feature_droplets_df)