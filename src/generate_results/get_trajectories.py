# import time
# import jax
# import jax.numpy as jnp
import numpy as np
import pandas as pd
# from tqdm import tqdm
import torch 

# import matplotlib.pyplot as plt
# from IPython import display

# from ott import utils
# from ott.geometry import costs, pointcloud
# from ott.problems.linear import linear_problem
# from ott.solvers.linear import sinkhorn
from pathlib import Path
import os

def transform_to_entry_based_probability_matrix(cfg, ot_matrix):
    """
    This function scales the OT matrix to a range of [0,1].
    """
    # Scale onto 0-1 range
    ot_matrix_scaled = (ot_matrix - ot_matrix.min()) / (ot_matrix.max() - ot_matrix.min())

    return ot_matrix_scaled

    ## VERSION 2 - more similar to rank based by using cutoff
    # # Scale onto 0-1 range
    # # Cut off the lowest 1% of the values
    # ot_m = np.array(ot_matrix)
   
    # # Standardize and scale the entries
    # ot_m = (ot_m - ot_m.min()) / (ot_m.max() - ot_m.min())

    # # Flatten the matrix into 1D array
    # flattened = np.array(ot_m).flatten()

    # # Get max rank per vector along maximal dimension
    # max_entry_per_vector = np.max(np.array(ot_m), axis=0)
    # cut_off = np.min(max_entry_per_vector)

    # # Get uncertainty resolution
    # uncertainty_resolution = cfg.generate_results.uncertainty_resolution

    # # Make sure to create contrast in probabilities where it matters
    # flattened[flattened <= cut_off / uncertainty_resolution] = cut_off / uncertainty_resolution
    
    # # 0-1 normalize the ranks
    # max_rank = np.max(flattened)
    # min_rank = np.min(flattened)

    # # Scale to 0-1 range
    # prob_matrix = (np.array(ot_m) - min_rank) / (max_rank - min_rank)

    # # Make sure that probabilities are uniformly distributed on [0,1]
    # prob_matrix = ot_m
    # prob_matrix[prob_matrix < 0] = 0

    # return prob_matrix

def transform_to_rank_based_probability_matrix(cfg, ot_matrix):
    """
    This function transforms the OT matrix to a probability matrix based on the ranks of the entries in the OT matrix.
    """
    # Flatten the matrix into 1D array
    flattened = np.array(ot_matrix).flatten()

    # Get the ordering of the elements
    ranks = flattened.argsort().argsort()

    # Reshape the ordering to match the original matrix shape
    ranks = ranks.reshape(ot_matrix.shape)

    # Make sure to create contrast in probabilities where it matters
    # max_dim = np.argmax(ot_matrix.shape)
    
    # Get max rank per vector along maximal dimension
    max_rank_per_vector = np.max(ranks, axis=0)
    min_max_rank_per_vector = np.min(max_rank_per_vector)

    # Get uncertainty resolution
    uncertainty_resolution = cfg.generate_results.uncertainty_resolution

    ranks[ranks <= min_max_rank_per_vector // uncertainty_resolution] = min_max_rank_per_vector // uncertainty_resolution - 1
    
    # 0-1 normalize the ranks
    max_rank = np.max(ranks)
    min_rank = np.min(ranks)
    prob_matrix = (ranks - min_rank) / (max_rank - min_rank)

    ## VERSION 2
    # # Flatten the matrix into 1D array
    # flattened = np.array(ot_matrix).flatten()

    # # Get the ordering of the elements
    # ranks = flattened.argsort().argsort()

    # # Reshape the ordering to match the original matrix shape
    # ranks = ranks.reshape(ot_matrix.shape)

    # # Make sure to create contrast in probabilities where it matters
    # max_dim = max(ot_matrix.shape[0], ot_matrix.shape[1])
    # min_dim = min(ot_matrix.shape[0], ot_matrix.shape[1])

    # # Get max rank of vectors of maximal dimension
    # max_rank = np.max(ranks, axis=max_dim-1)
    
    # uncertainty_resolution = cfg.generate_results.uncertainty_resolution
    # ranks[ranks <= max_dim * (min_dim - uncertainty_resolution)] = max_dim * (min_dim - uncertainty_resolution)
    
    # # 0-1 normalize the ranks
    # max_rank = np.max(ranks)
    # min_rank = np.min(ranks)
    # prob_matrix = (ranks - min_rank) / (max_rank - min_rank)

    ## VERSION 1
    # # Flatten the matrix into 1D array
    # flattened = ot_matrix.flatten()

    # # Get the ordering of the elements
    # ranks = flattened.argsort().argsort()

    # # Reshape the ordering to match the original matrix shape
    # ranks = ranks.reshape(ot_matrix.shape)

    # # Make sure the the maximal probability is 1
    # min_dim = min(ot_matrix.shape[0], ot_matrix.shape[1])
    # ranks = ranks // min_dim

    # # Add row and col based ordering to deal with ties
    # row_ranks = ot_matrix.argsort().argsort()
    # col_ranks = ot_matrix.argsort(axis=0).argsort(axis=0)
    # ranks_row_col = row_ranks + col_ranks
    # norm_ranks_row_col = ranks_row_col / (ot_matrix.shape[0] + ot_matrix.shape[1] - 2)

    # # Transform to probability matrix
    # total_ranks = ranks + norm_ranks_row_col

    # # Apply temperature for more constrast in the probabilities
    # temp = 2
    # prob_matrix = total_ranks ** temp / (min_dim+1) ** temp

    return prob_matrix



def get_epsilon_independent_id_mapping(cfg, ot_matrix, frame_id):
    """
    This function returns a mapping from the droplet ids in the current frame to the droplet ids in the next frame.
    The mapping is epsilon independent and based on ranks.
    """
    # Get directory to store probability matrices
    directory_name = "prob_matrix_" + cfg.experiment_name
    directory_path = Path(cfg.data_path) / Path(cfg.ot_dir) / Path(cfg.experiment_name) / Path(directory_name)
    name = f"{frame_id}-{frame_id+1}.npy"

    # Get probability matrix based on ranks
    if cfg.generate_results.uncertainty_type == "scaled_ranks":
        prob_matrix = transform_to_rank_based_probability_matrix(cfg, ot_matrix)

    elif cfg.generate_results.uncertainty_type == "scaled_entries":
        prob_matrix = transform_to_entry_based_probability_matrix(cfg, ot_matrix)

    else:
        raise NotImplementedError("Unknown uncertainty type. In get_trajectories.py.")
    
    # Store the probability matrix
    torch.save(prob_matrix, directory_path / name)

    # Get choices of algorithm - choose to prioritize finding a trajectory for the droplet in the current frame
    next_frame_ids = np.argmax(ot_matrix, axis=1)
    this_frame_ids = np.arange(ot_matrix.shape[0])

    return this_frame_ids, next_frame_ids

def create_trajectory_with_prob(cfg, ot_matrices):
    """
    This function creates a tracking table with the droplet ids of the current frame and the next frame.
    The transitions are based on transition scores extracted from the OT matrices.
    """
    tracking_table = []

    for i, ot_matrix in enumerate(ot_matrices):
        # get the mapping from the current frame to the next frame
        this_frame_ids, next_frame_ids = get_epsilon_independent_id_mapping(cfg, ot_matrix, i)

        # create a dataframe with the indices and the probabilities
        tmp = pd.DataFrame({f'frame': i,
                            f'droplet_id_this': this_frame_ids,
                            f'droplet_id_next': next_frame_ids})
        
        # append the dataframe to the tracking table
        tracking_table.append(tmp)

    tracking_table = pd.concat(tracking_table, axis=0)

    return tracking_table 

def filter_and_reindex_droplets(cfg, droplet_table: pd.DataFrame, frame_id: int, reindex=True) -> pd.DataFrame:
    """
    This function filters and reindexes the droplets that are present in the current frame.
    Filtering only needs to be done in unbalanced evaluation mode.
    """
    # Get the droplets that are present in the current frame
    droplets = droplet_table[droplet_table['frame'] == frame_id]

    # Filter the droplets that are cut out to imitate the unbalanced real data
    if cfg.generate_results.evaluation_mode == True and cfg.evaluate.cutout_image == True:
        droplets = droplets[droplets['center_x'] > cfg.evaluate.cutout.x_min]
        droplets = droplets[droplets['center_x'] < cfg.evaluate.cutout.x_max]
        droplets = droplets[droplets['center_y'] > cfg.evaluate.cutout.y_min]
        droplets = droplets[droplets['center_y'] < cfg.evaluate.cutout.y_max]

    # Reindex the droplets
    if reindex:
        droplets["droplet_id"] = np.arange(len(droplets))

    return droplets

def process_and_merge_results(cfg, droplet_table: pd.DataFrame, 
                        tracking_table: pd.DataFrame) -> pd.DataFrame:
    """
    Create trajectories from the tracking table and the droplet table.
    The following will be stored per transition:
    - droplet_ids
    - x and y positions
    - probabilities
    - number of cells
    """
    # Get droplet IDs that we want to track. These are the droplets that are present in the first frame.
    droplets_raw = [df for _, df in droplet_table.groupby('frame', sort=True)]
    original_ids = droplets_raw[0]['droplet_id'].to_numpy()

    # To make trajectories comparable, we reindex the droplets in each frame
    droplets = [filter_and_reindex_droplets(cfg,df,frame) for frame,df in enumerate(droplets_raw)]
    original_droplets = [filter_and_reindex_droplets(cfg,df,frame, reindex=False) for frame,df in enumerate(droplets_raw)]

    # Get number of droplets
    num_droplets = len(droplets[0]['droplet_id'])
    droplets_first_frame = np.arange(num_droplets)

    # Create result dataframe to fill
    result = pd.DataFrame({'reindexed_droplet_id': droplets_first_frame}, dtype=int)
     
    # Add original ids
    result['id_0'] = original_ids

    # Get the first positions of the droplets
    result = result.merge(droplets[0], left_on='reindexed_droplet_id', right_on='droplet_id', how='left').drop(columns=['frame','radius'])

    # Rename columns and drop droplet_ids
    result = result.rename(columns={'center_x': 'x0', 
                                    'center_y': 'y0', 
                                    'nr_cells': 'nr_cells0'}).drop(columns=['reindexed_droplet_id','droplet_id'])
    
    # Add starting probabilities of 1
    full_prob = np.ones((droplets_first_frame.shape[0],), dtype=np.float64)
    
    # Define types
    result = result.astype({'x0': float, 'y0': float,'nr_cells0': 'Int32'})

    # Id column for merging
    curr_droplet_ids = droplets_first_frame.copy()

    # Iterate through frames
    for i, track_df in tracking_table.groupby('frame', sort=True):
        # Get the next positions of the droplets
        next_droplets = droplets[i+1]
        next_droplets_original = original_droplets[i+1]

        # Get the droplet locations for the next frame and transition probabilities
        next_ids = track_df['droplet_id_next'].to_numpy()

        # Get current transition matrix
        directory_name = "prob_matrix_" + cfg.experiment_name
        directory_path = Path(cfg.data_path) / Path(cfg.ot_dir) / Path(cfg.experiment_name) / Path(directory_name)
        name = f"{i}-{i+1}.npy"
        prob_matrix = torch.load(directory_path / name)

        # Maintain mapping of droplets
        droplet_ids_next = [next_droplets["droplet_id"].to_numpy()[next_ids[i]] for i in curr_droplet_ids]
        droplet_ids_original_next = [next_droplets_original["droplet_id"].to_numpy()[next_ids[i]] for i in curr_droplet_ids]

        # Get probabilities of chosen trajectory
        probs = np.array([prob_matrix[curr_droplet_ids[i], droplet_ids_next[i]] for i in range(len(curr_droplet_ids))])
        
        # Store for next iteration
        curr_droplet_ids = droplet_ids_next.copy()

        # Get the next positions and metadata of the droplets
        next_x = np.array([next_droplets['center_x'].to_numpy()[droplet_idx] for droplet_idx in droplet_ids_next])
        next_y = np.array([next_droplets['center_y'].to_numpy()[droplet_idx] for droplet_idx in droplet_ids_next])
        next_num_cells = np.array([next_droplets['nr_cells'].to_numpy()[droplet_idx] for droplet_idx in droplet_ids_next])
        
        # Aggregate probabilities 
        if cfg.generate_results.prob_agregation == "multiplication":
            full_prob = full_prob * probs

        elif cfg.generate_results.prob_agregation == "average":
            # Running average of probabilities
            if i == 0:
                full_prob = probs
            else:
                full_prob = (i/(i+1))*full_prob + (1/(i+1))*probs
            full_prob[probs == 0.0] = 0.0

        # Add columns to dataframe
        result[f'id_{i+1}'] = droplet_ids_original_next
        result[f'x{i+1}'] = next_x
        result[f'y{i+1}'] = next_y
        result[f'p{i}_{i+1}'] = probs
        result[f'nr_cells{i+1}'] = next_num_cells
        result = result.astype({f'x{i+1}': float, f'y{i+1}': float,f'nr_cells{i+1}': 'Int32'})

        # Set the probability of transitions that are not possible to zero
        # get rows where there are no entries for x, y positions
        idx = result[f'x{i+1}'].isnull()
        idx = idx | result[f'x{i}'].isnull()

        # set the probability to zero
        result.loc[idx, f'p{i}_{i+1}'] = 0.0
    
    # Add final probability
    result['p'] = full_prob

    return result


def part_trajectory_prob(cfg, df):
    """
    Computes probabilities of partial trajectories.
    """
    probs = []
    for col in df.columns.sort_values():
        if col[0] == 'p' and col != 'p':
            probs.append(col)
    df = df[probs]

    # Calculate the product in sliding windows
    num_columns = df.shape[1]

    if cfg.generate_results.prob_agregation == "multiplication":
        sliding_windows = [np.prod(df.iloc[:, i:j], axis=1) for i in range(num_columns) for j in range(i + 2, num_columns + 1)]
    
    elif cfg.generate_results.prob_agregation == "average":
        sliding_windows = [np.mean(df.iloc[:, i:j], axis=1) for i in range(num_columns) for j in range(i + 2, num_columns + 1)]

    else:
        raise NotImplementedError("Unknown probability aggregation method.")
    
    # Create a new DataFrame with the results
    result_df = pd.DataFrame({
        f'p{i}-{j}': product
        for (i, j), product in zip(
            [(i, j) for i in range(num_columns) for j in range(i + 2, num_columns + 1)],
            sliding_windows
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

    # reorder the columns
    cols = final_results_df.columns.tolist()
    locs = [col for col in cols if col.startswith("x") or col.startswith("y")]
    probs = [col for col in cols if col.startswith("p") and col != "p"]
    num_cells = [col for col in cols if col.startswith("nr_cells")]
    original_ids = [col for col in cols if col.startswith("id")]
    new_cols = ["p"] + original_ids + locs + probs + num_cells
    final_results_df = final_results_df.reindex(columns=new_cols)
    final_results_df = final_results_df.rename(columns={"p": "full_trajectory_uncertainty"})

    # Round numerical values
    final_results_df = final_results_df.round(3)

    # save the results
    final_results_df.to_csv(image_results_path / f'results_{cut_name}.csv', index=False)


def compute_and_store_results_all(cfg, image_ot_path, image_results_path, image_feature_path):
    # Progress
    if cfg.verbose:
        print("\n=========================================")
        print("Generating Results For all Cuts")
        print("=========================================\n")

        print(f'Currently generating results for cut:')
    
    # Create directories for storing probability interpretation of ot matrix
    # Create directory if it does not exist
    directory_name = "prob_matrix_" + cfg.experiment_name
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

#################### OLD ####################
# def process_and_merge_results_old(cfg, droplet_table: pd.DataFrame, 
#                         tracking_table: pd.DataFrame) -> pd.DataFrame:
#     droplets = [df for _, df in droplet_table.groupby('frame', sort=True)]
#     max_droplets = droplet_table['droplet_id'].max() + 1
#     full_prob = np.ones((max_droplets,), dtype=np.float64)

#     result = pd.DataFrame({'droplet_id': np.arange(max_droplets)}, dtype=int)

#     result = result.merge(droplets[0], left_on='droplet_id', right_on='droplet_id', how='left').drop(columns='frame')
#     result = result.rename(columns={'center_x': f'x0', 
#                                     'center_y': f'y0', 
#                                     'radius': f'r0',
#                                     'nr_cells': f'nr_cells0'})
#     result['droplet_id_next'] = pd.DataFrame({'droplet_id': np.arange(max_droplets)}, dtype=int)
#     result = result.astype({'x0': float, 'y0': float,'r0': 'Int32','nr_cells0': 'Int32'})
#     for i, df in tracking_table.groupby('frame'):
#         next = droplets[i+1]

#         result = result.merge(df[['droplet_id_this', 'prob', 'droplet_id_next']], left_on='droplet_id_next', right_on='droplet_id_this', how='left')

#         probs = result['prob'].fillna(0.0).to_numpy()


#         result = result.rename(columns={'prob': f'p{i}_{i+1}','droplet_id_next_y': 'droplet_id_next'}).drop(columns=['droplet_id_this','droplet_id_next_x',])
#         result = result.merge(next, left_on='droplet_id_next', right_on='droplet_id', how='left').drop(columns='frame')
#         result = result.rename(columns={'droplet_id_x': 'droplet_id',
#                                         'center_x': f'x{i+1}', 
#                                         'center_y': f'y{i+1}',
#                                         'radius': f'r{i+1}',
#                                         'nr_cells': f'nr_cells{i+1}'}).drop(columns='droplet_id_y')
#         result = result.astype({f'x{i+1}': float, f'y{i+1}': float, f'r{i+1}': 'Int32',f'nr_cells{i+1}': 'Int32'})

#         # Set the probability of transitions that are not possible to zero
#         # get rows where there are no entries for x, y positions
#         idx = result[f'x{i+1}'].isnull()
#         idx = idx | result[f'x{i}'].isnull()

#         # set the probability to zero
#         result.loc[idx, f'p{i}_{i+1}'] = 0.0
        
#         if cfg.generate_results.prob_agregation == "multiplication":
#             full_prob = full_prob * probs

#         elif cfg.generate_results.prob_agregation == "average":
#             # Running average of probabilities
#             full_prob = ((i+1)/(i+2))*full_prob + (1/(i+2))*probs
#             full_prob[probs == 0.0] = 0.0
        
#     result['p'] = full_prob
#     result.insert(1, 'full_trajectory_uncertainty', full_prob, allow_duplicates=True)
#     result = result.drop(columns=['droplet_id_next'])

#     return result

# def get_indices_to_drop(cfg, probs, vec):
#     """
#     This function returns the indices of the entries that should be dropped.
#     Indices are dropped if they have a duplicate entry in vec and the probability is lower than the maximum probability.
#     """
#     # Find duplicate entries in next_frame_ids
#     unique, counts = np.unique(vec, return_counts=True)
#     duplicates = unique[counts > 1]

#     # Find the indices of the duplicates
#     idxs_to_remove = []
#     probs_np = np.array(probs)
#     for duplicate in duplicates:
#         # Find the indices of the duplicates
#         indices = np.where(vec == duplicate)[0]

#         # Find the index of the maximum probability
#         max_index = indices[probs_np[indices].argmax()]

#         # Remove the all duplicates with the exception of the maximum probability
#         idxs_to_remove.extend(indices[indices != max_index])

#     return idxs_to_remove

# def get_epsilon_dependent_id_mapping_sven(cfg, ot_matrix):
#     """
#     This function returns a mapping from the droplet ids in the current frame to the droplet ids in the next frame 
#     based on the entries of the OT matrix. The mapping is epsilon dependent.
#     """
#     threshold = 1/(ot_matrix.shape[0]*ot_matrix.shape[1])**2

#     # masking entries that are below the threshold as True
#     mask_threshold = ot_matrix < threshold

#     rows_with_all_entries_less_threshold = np.all(mask_threshold, axis=1)
#     cols_with_all_entries_less_threshold = np.all(mask_threshold, axis=0)
    
#     # copy the ot matrix and set all values to zero that are below the threshold
#     prob = ot_matrix.copy()
#     prob = prob.at[mask_threshold].set(0.0)

#     # set all columns to zero that have no connection (droplet is not in the current frame)
#     prob = prob.at[:, cols_with_all_entries_less_threshold].set(0.0)

#     # make row sum to 1 (to get probabilities)
#     prob = prob/prob.sum(axis=1)[:, None]

#     # mask rows that have no connection (droplet is not in the next frame)
#     prob = prob.at[rows_with_all_entries_less_threshold, :].set(0.0)

#     # only keep values that are positive
#     ids_prob = prob > 0.0

#     # combine the two masks
#     ids_max_this = prob == prob.max(axis=1)[:, None]
#     ids_max_next = prob == prob.max(axis=0)[None, :]
    
#     ids_max = np.logical_and(ids_max_this, ids_max_next)
#     ids = np.logical_and(ids_prob, ids_max)
    
#     #######
#     ids[np.where(ids_max_this.sum(axis=1) > 1), :] = False
#     ids[:, np.where(ids_max_next.sum(axis=0) > 1)] = False
#     #######

#     this_frame_ids, next_frame_ids = np.where(ids)

#     probs = prob[this_frame_ids, next_frame_ids]

#     return this_frame_ids, next_frame_ids, probs