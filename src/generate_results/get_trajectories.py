# Types and os
from pathlib import Path
from typing import Tuple
import os

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
import torch 
from joblib import load

def transform_to_entry_based_probability_matrix(cfg, ot_matrix: np.ndarray) -> np.ndarray:
    """
    Linearly scale entries of OT matrix entries onto the range of [0,1].

    Parameters:
    ----------
    cfg : Configuration
        The configuration object.
    ot_matrix : np.ndarray
        The OT matrix to be scaled.

    Returns:
    -------
    ot_matrix_scaled : np.ndarray
        The min-max scaled OT matrix.
    """
    # Scale onto 0-1 range
    ot_matrix_scaled = (ot_matrix - ot_matrix.min()) / (ot_matrix.max() - ot_matrix.min())

    return ot_matrix_scaled


def transform_to_rank_based_probability_matrix(cfg, ot_matrix: np.ndarray) -> np.ndarray:
    """
    Transform the OT matrix to a probability matrix based on the ranks of the entries in the OT matrix.

    Parameters:
    ----------
    cfg : Configuration
        The configuration object containing the settings for generating results.
    ot_matrix : np.ndarray
        The OT matrix to be transformed.

    Returns:
    -------
    prob_matrix : np.ndarray
        The transformed probability matrix.
    """
    # Flatten the matrix into 1D array
    flattened = np.array(ot_matrix).flatten()

    # Get the ordering of the elements
    ranks = flattened.argsort().argsort()

    # Reshape the ordering to match the original matrix shape
    ranks = ranks.reshape(ot_matrix.shape)
    
    # Get max rank per vector along maximal dimension
    max_rank_per_vector = np.max(ranks, axis=0)

    # 0-1 normalize the ranks
    max_rank = np.max(ranks)
    min_rank = np.min(ranks)
    prob_matrix = (ranks - min_rank) / (max_rank - min_rank)

    return prob_matrix


def get_id_mapping(cfg, ot_matrix: np.ndarray, 
                   frame_id: np.ndarray, 
                   cut_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a mapping from droplet ids in the current frame to the droplet ids in the next frame.

    Parameters:
    ----------
    cfg : Configuration
        The configuration object.
    ot_matrix : np.ndarray
        The optimal transport matrix.
    frame_id : np.ndarray
        The id of the current frame.
    cut_name : str
        The name of the cut.

    Returns:
    -------
    this_frame_ids : np.ndarray
        The droplet ids in the current frame.
    next_frame_ids : np.ndarray
        The droplet ids in the next frame.
    """
    # Get directory to store probability matrices
    directory_name = "prob_matrix" + cut_name
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


def create_trajectory_with_prob(cfg, ot_matrices: list, cut_name: str) -> pd.DataFrame:
    """
    This function creates a tracking table with the droplet ids of the current frame and the next frame.
    The transitions are based on transition scores extracted from the OT matrices.

    Parameters:
    ----------
    cfg : Configuration
        The configuration object.
    ot_matrices : list
        A list of OT matrices.

    Returns:
    -------
    tracking_table : pd.DataFrame
        A dataframe with the droplet ids of the current frame and the next frame.
    """
    tracking_table = []

    for i, ot_matrix in enumerate(ot_matrices):
        # get the mapping from the current frame to the next frame
        this_frame_ids, next_frame_ids = get_id_mapping(cfg, ot_matrix, i, cut_name)

        # Make sure that frame range is considered
        if cfg.generate_results.frame_range is not None:
            first_frame=cfg.generate_results.frame_range[0]
            i += first_frame

        # create a dataframe with the indices and the probabilities
        tmp = pd.DataFrame({f'frame': i,
                            f'droplet_id_this': this_frame_ids,
                            f'droplet_id_next': next_frame_ids})
        
        # append the dataframe to the tracking table
        tracking_table.append(tmp)

    tracking_table = pd.concat(tracking_table, axis=0)

    return tracking_table 


def filter_and_reindex_droplets(cfg, 
                                droplet_table: pd.DataFrame, 
                                frame_id: int, 
                                reindex: bool = True) -> pd.DataFrame:
    """
    This function filters and reindexes the droplets that are present in the current frame.
    Filtering only needs to be done in unbalanced evaluation mode.

    Parameters:
    ----------
    cfg : Configuration
        The configuration object.
    droplet_table : pd.DataFrame
        The droplet table.
    frame_id : int
        The id of the current frame.
    reindex : bool
        Whether to reindex the droplets.

    Returns:
    -------
    droplets : pd.DataFrame
        The filtered and reindexed droplets.
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


def process_and_merge_results(cfg, 
                              droplet_table: pd.DataFrame, 
                              tracking_table: pd.DataFrame,
                              cut_name: str) -> pd.DataFrame:
    """
    Create trajectories from the tracking table and the droplet table.
    The following will be stored per transition:
    - droplet_ids
    - x and y positions
    - probabilities
    - number of cells

    Parameters:
    ----------
    cfg : Configuration
        The configuration object.
    droplet_table : pd.DataFrame
        The droplet table.
    tracking_table : pd.DataFrame
        The tracking table.
    cut_name : str
        The name of the cut.

    Returns:
    -------
    result : pd.DataFrame
        The dataframe with the trajectories.
    """
    # Get droplet IDs that we want to track. These are the droplets that are present in the first frame.
    droplets_raw = [df for _, df in droplet_table.groupby('frame', sort=True)]

    # To make trajectories comparable, we reindex the droplets in each frame
    droplets = [filter_and_reindex_droplets(cfg,df,frame) for frame,df in enumerate(droplets_raw)]
    original_droplets = [filter_and_reindex_droplets(cfg,df,frame, reindex=False) for frame,df in enumerate(droplets_raw)]
     
    # Add original ids and make sure frame range is considered
    if cfg.generate_results.frame_range is not None:
        first_frame=cfg.generate_results.frame_range[0]
        last_frame=cfg.generate_results.frame_range[1]
    else:
        first_frame = 0
        last_frame = len(droplets)-1

    # Get number of droplets
    num_droplets = len(droplets[first_frame]['droplet_id'])

    if cfg.generate_results.evaluation_mode:
        # Find number of droplets in first frame (different from droplet.csv in evaluation mode)
        for i, track_df in tracking_table.groupby('frame', sort=True):
            num_droplets = len(track_df)
            break

    droplets_first_frame = np.arange(num_droplets)
    original_ids = droplets_raw[first_frame]['droplet_id'].to_numpy()

    # Create result dataframe to fill
    result = pd.DataFrame({'reindexed_droplet_id': droplets_first_frame}, dtype=int)
    col_name = f'id_{first_frame}'

    if cfg.generate_results.evaluation_mode:
        result[col_name] = original_ids[:num_droplets]
    else:
        result[col_name] = original_ids


    # Get the first positions of the droplets
    result = result.merge(droplets[first_frame], left_on='reindexed_droplet_id', right_on='droplet_id', how='left').drop(columns=['frame','radius'])

    # Rename columns and drop droplet_ids
    result = result.rename(columns={'center_x': f'x{first_frame}', 
                                    'center_y': f'y{first_frame}', 
                                    'nr_cells': f'nr_cells{first_frame}'}).drop(columns=['reindexed_droplet_id','droplet_id'])
    
    # Add starting probabilities of 1
    full_prob = np.ones((droplets_first_frame.shape[0],), dtype=np.float64)
    
    # Define types
    result = result.astype({f'x{first_frame}': float, f'y{first_frame}': float, f'nr_cells{first_frame}': 'Int32'})

    # Id column for merging
    curr_droplet_ids = droplets_first_frame.copy()

    # Iterate through frames
    for i, track_df in tracking_table.groupby('frame', sort=True):
        # Skip frames out of range
        if i < first_frame or i >= last_frame:
            continue

        # Get the next positions of the droplets
        next_droplets = droplets[i+1]
        next_droplets_original = original_droplets[i+1]

        # Get the droplet locations for the next frame and transition probabilities
        next_ids = track_df['droplet_id_next'].to_numpy()

        # Get current transition matrix
        directory_name = "prob_matrix" + cut_name
        directory_path = Path(cfg.data_path) / Path(cfg.ot_dir) / Path(cfg.experiment_name) / Path(directory_name)
        name = f"{i}-{i+1}.npy"
        prob_matrix = torch.load(directory_path / name)

        # Maintain mapping of droplets
        droplet_ids_next = [next_droplets["droplet_id"].to_numpy()[next_ids[i]] for i in curr_droplet_ids]
        droplet_ids_original_next = [next_droplets_original["droplet_id"].to_numpy()[next_ids[i]] for i in curr_droplet_ids]

        # Get probabilities of chosen trajectory
        probs = np.array([prob_matrix[curr_droplet_ids[i], droplet_ids_next[i]] for i in range(len(curr_droplet_ids))])
        
        # Calibrate probabilities if desired
        if cfg.generate_results.calibrate_probabilities:
            # Load calibration model
            calibration_model_path = Path(cfg.calibration_model_dir)
            calibration_model = load(Path(calibration_model_path / cfg.generate_results.calibration_model_name))

            # Apply calibration model
            probs = calibration_model.transform(probs)

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


def part_trajectory_prob(cfg, df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes probabilities of partial trajectories.

    Parameters:
    ----------
    cfg : Configuration
        The configuration object.
    df : pd.DataFrame
        The dataframe with the trajectories.

    Returns:
    -------
    result_df : pd.DataFrame
        The dataframe with the partial trajectory probabilities.
    """
    probs = []
    for col in df.columns.sort_values():
        if col[0] == 'p' and col != 'p':
            probs.append(col)
    df = df[probs]

    # Aggregate in sliding windows
    num_columns = df.shape[1]
    
    # Choose type of aggregation
    if cfg.generate_results.prob_agregation == "multiplication":
        sliding_windows = [np.prod(df.iloc[:, i:j], axis=1) for i in range(num_columns) for j in range(i + 2, num_columns + 1)]
    
    elif cfg.generate_results.prob_agregation == "average":
        sliding_windows = [np.mean(df.iloc[:, i:j], axis=1) for i in range(num_columns) for j in range(i + 2, num_columns + 1)]

    else:
        raise NotImplementedError("Unknown probability aggregation method.")
    
    # Create a new DataFrame with the results
    if cfg.generate_results.frame_range is not None:
        first_frame = cfg.generate_results.frame_range[0]
    else:
        first_frame = 0

    result_df = pd.DataFrame({
        f'p{i}-{j}': product
        for (i, j), product in zip(
            [(i+first_frame, j+first_frame) for i in range(num_columns) for j in range(i + 2, num_columns + 1)],
            sliding_windows
        )
    })

    # Display the result DataFrame
    return result_df


def filter_results(cfg, results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter the results based on the following criteria:
    - Confidence threshold on whole trajectory
    - Duplicate assignments

    Optionally also returns the trajectories that are dropped due to merging as a dataframe.

    Parameters:
    ----------
    cfg : Configuration
        The configuration object.
    results_df : pd.DataFrame
        The dataframe with the trajectories.

    Returns:
    -------
    trajectories : pd.DataFrame
        The filtered dataframe with the trajectories.
    dropped_merging_trajectories : pd.DataFrame
        The dataframe with the dropped merging trajectories.
    """
    # Get copy of results
    trajectories = results_df.copy()

    # Params from config
    uncertainty_threshold = cfg.filter_results.uncertainty_threshold
    enforce_same_number_of_cells = cfg.filter_results.enforce_same_nr_cells
    max_distance = cfg.filter_results.max_distance
    frame_margin = cfg.filter_results.frame_margin
    filter_merging_trajectories = cfg.filter_results.filter_merging_trajectories

    # Get range of frames
    if cfg.filter_results.frame_range is None:
        first_frame = 0
        cols = trajectories.columns.tolist()
        original_ids = [int(col.split("_")[1]) for col in cols if col.startswith("id")]
        last_frame = max(original_ids)
    
    else:
        first_frame = cfg.filter_results.frame_range[0]
        last_frame = cfg.filter_results.frame_range[1]

    # Test for full trajectory uncertainty passing the threshold
    if uncertainty_threshold is not None:
        mask = trajectories[f'full_trajectory_uncertainty'] >= uncertainty_threshold
        trajectories = trajectories[mask]

    # Test for the same number of cells
    if enforce_same_number_of_cells:
        mask = pd.Series(True, index=trajectories.index)
        for i in range(first_frame + 1, last_frame + 1):
            mask = mask & (trajectories[f'nr_cells{i}'] == trajectories[f'nr_cells{first_frame}'])

        trajectories = trajectories[mask]

    # Test for max distance between frames
    if max_distance is not None:
        mask = pd.Series(True, index=trajectories.index)
        for i in range(first_frame, last_frame):
            mask = mask & ((trajectories[f'x{i}'] - trajectories[f'x{i + 1}']).abs() < max_distance)
            mask = mask & ((trajectories[f'y{i}'] - trajectories[f'y{i + 1}']).abs() < max_distance)

        trajectories = trajectories[mask]

    # Test for distance from the frame border
    if frame_margin is not None:
        min_x = 0
        max_x = trajectories[f'x{first_frame}'].max()
        min_y = 0
        max_y = trajectories[f'y{first_frame}'].max()
        mask = pd.Series(True, index=trajectories.index)
        for i in range(first_frame, last_frame + 1):
            mask = mask & (trajectories[f'x{i}'] - min_x > frame_margin)
            mask = mask & (max_x - trajectories[f'x{i}'] > frame_margin)
            mask = mask & (trajectories[f'y{i}'] - min_y > frame_margin)
            mask = mask & (max_y - trajectories[f'y{i}'] > frame_margin)
        
        trajectories = trajectories[mask]

    # Filter for merging trajectories and keep only the ones with the highest probability
    if filter_merging_trajectories:
        # Sort the trajectories by their uncertainty
        trajectories = trajectories.sort_values(by="full_trajectory_uncertainty", ascending=False)

        # Iterate through frame transitions and filter for merging trajectories
        cols = trajectories.columns.tolist()
        original_ids = [col for col in cols if col.startswith("id")]

        # Filter for frames before and after the frame range
        original_ids = [col for col in original_ids if int(col.split("_")[1]) >= first_frame and int(col.split("_")[1]) <= last_frame]

        trajectories_high_prob = trajectories.copy()

        for col in original_ids:
            trajectories_high_prob = trajectories_high_prob.drop_duplicates(col, keep="first")
        
        # Get the indices of the trajectories that are not kept
        mask = trajectories.index.isin(trajectories_high_prob.index)
        trajectories_low_prob = trajectories[~mask]

        return trajectories_high_prob, trajectories_low_prob
    
    return trajectories


def compute_mvt_metric(cfg, result_df: pd.DataFrame, normalize: bool = True, droplet_radius: float = 0, return_dist: bool = False) -> pd.DataFrame:
    """
    Compute mvt metrics for the predicted trajectories.
    
    Parameters:
    ----------
    cfg : Configuration
        The configuration object.
    result_df : pd.DataFrame
        The dataframe with the trajectories.
    normalize : bool
        Whether to normalize the distances by the maximal possible distance.
    
    Returns:
    -------
    mvts : pd.DataFrame
        The dataframe with the mvt metric.
    """
    # Extract the columns containing positional information
    cols_to_drop = [col for col in result_df.columns if not col.startswith('x') and not col.startswith('y')]
    result_df = result_df.drop(columns=cols_to_drop)
    start_frame = np.min([int(col[1:]) for col in result_df.columns])

    # Get number of frames
    n_frames = result_df.columns.size // 2

    # Compute the distance between each frame
    mvts = pd.DataFrame()
    if normalize == "droplet_radius":
        mvts.index = ["mean", "std", "median", "mad", "min", "max", "q20", "q80", "p_moved_1r", "p_moved_5r"]
    else:
        mvts.index = ["mean", "std", "median", "mad", "min", "max", "q20", "q80"]

    # Compute the distance traveled by each droplet between each frame
    for i in range(start_frame, start_frame+n_frames-1):
        # Compute distances
        x_diff = result_df['x'+str(i+1)] - result_df['x'+str(i)]
        y_diff = result_df['y'+str(i+1)] - result_df['y'+str(i)]
        distances = (x_diff**2 + y_diff**2)**0.5

        # Normalize distances by the maximal possible distance
        if normalize is not None:
            if normalize == "max_distance":
                max_dist = (result_df['x'+str(i+1)].max() - result_df['x'+str(i)].min())**2 + (result_df['y'+str(i+1)].max() - result_df['y'+str(i)].min())**2
                distances = distances / max_dist * 1000 # This way the value indicates a percentage of the maximal possible distance
            elif normalize == "droplet_radius":
                # TODO: Extract droplet radius from droplet detection step
                distances = distances / droplet_radius
                p_moved_1r = (distances > 1).sum() / distances.size
                p_moved_5r = (distances > 5).sum() / distances.size
            else:
                raise ValueError(f"Unknown normalization method: {normalize}")

        # Compute mean, std, min, max
        mean = distances.mean()
        std = distances.std()
        median = distances.median()
        mad = median_abs_deviation(distances)
        min = distances.min()
        max = distances.max()
        
        # drop the 5% of the largest values
        p20 = distances.quantile(0.20)
        p80 = distances.quantile(0.80)

        # Store results
        if normalize == "droplet_radius":
            mvts[f'{i}_{i+1}'] = [mean, std, median, mad, min, max, p20, p80, p_moved_1r, p_moved_5r]
        else:
            mvts[f'{i}_{i+1}'] = [mean, std, median, mad, min, max, p20, p80]

    # Obtain metrics aggregated over all frames
    mvts['all_mean'] = mvts.mean(axis=1)
    mvts['all_std'] = mvts.std(axis=1)

    # Put the last two columns in front
    cols = mvts.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    mvts = mvts[cols]

    # Round numerical values
    mvts = mvts.round(3)
    
    if return_dist:
        return mvts, distances
    else:
        return mvts


def compute_and_store_results_cut(cfg, 
                                  cut_name: str, 
                                  cut_ot_path: Path, 
                                  image_results_path: Path, 
                                  cut_feature_droplets_df: pd.DataFrame) -> None:
    """
    Compute and store the results for a single cut.

    Parameters:
    ----------
    cfg : Configuration
        The configuration object.
    cut_name : str
        The name of the cut.
    cut_ot_path : Path
        The path to the ot matrices of the cut.
    image_results_path : Path
        The path to the results of the cut.
    cut_feature_droplets_df : pd.DataFrame
        The dataframe with the droplet features of the cut.
    """
    # Frames to get results for
    if cfg.generate_results.frame_range is not None:
        first_frame = cfg.generate_results.frame_range[0]
        last_frame = cfg.generate_results.frame_range[1]

    # load the ot matrices
    ot_matrices = []
    for file_name in sorted(os.listdir(cut_ot_path), key=lambda x: int(x.split("-")[0])):
        if file_name.endswith(".npy"):
            frame_id = int(file_name.split("-")[0])
            if cfg.generate_results.frame_range is not None:
                if frame_id < first_frame or frame_id >= last_frame:
                    continue
            ot_matrices.append(torch.load(cut_ot_path / file_name))
    
    # create directory for probability matrices
    # Get file names for current cut
    cut_prob_path = str(cut_ot_path).replace("ot_matrix", "prob_matrix")

    if not os.path.exists(cut_prob_path):
        os.makedirs(cut_prob_path)

    # create the trajectory dataframe
    trajectory_df = create_trajectory_with_prob(cfg, ot_matrices, cut_name)

    # process the results
    results_df = process_and_merge_results(cfg, cut_feature_droplets_df, trajectory_df, cut_name)
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

    # Compute mvt metric
    # Get droplet radius
    if cfg.generate_results.droplet_radius is not None:
        droplet_radius = cfg.generate_results.droplet_radius
    else:
        droplet_radius = cut_feature_droplets_df['radius'].mean()
    final_results_df_mvt = compute_mvt_metric(cfg, final_results_df, normalize=cfg.generate_results.normalize_mvt, droplet_radius=droplet_radius)

    # save the results
    final_results_df.to_csv(image_results_path / f'unfiltered_trajectories{cut_name}.csv', index=False)
    final_results_df_mvt.to_csv(image_results_path / f'mvt_unfiltered_trajectories{cut_name}.csv', index=True)

    # filter the results
    if cfg.filter_results.filter_merging_trajectories:
        filtered_final_results, dropped_merging_trajectories_ = filter_results(cfg, final_results_df)
        dropped_merging_trajectories_.to_csv(image_results_path / Path(f'dropped_merging_trajectories{cut_name}' + cfg.filter_results.file_name_suffix + '.csv'), index=False)

    else:
        filtered_final_results = filter_results(cfg, final_results_df)

    # Compute mvt metric for filtered trajectories
    #filtered_final_results_mvt = compute_mvt_metric(cfg, filtered_final_results, normalize=True)

    # Store filtered results
    filtered_final_results.to_csv(image_results_path / Path(f'filtered_trajectories{cut_name}' + cfg.filter_results.file_name_suffix + '.csv'), index=False)

    # Also store all trajectories not in the filtered results
    mask = final_results_df.index.isin(filtered_final_results.index)
    dropped_trajectories = final_results_df[~mask]

    # Compute mvt metric for dropped trajectories
    #dropped_trajectories_mvt = compute_mvt_metric(cfg, dropped_trajectories, normalize=True)

    dropped_trajectories.to_csv(image_results_path / Path(f'dropped_trajectories{cut_name}' + cfg.filter_results.file_name_suffix + '.csv'), index=False)


def compute_and_store_results_all(cfg, image_ot_path, image_results_path, image_feature_path):
    # Progress
    if cfg.verbose:
        print("\n=========================================")
        print("Generating Results For all Cuts")
        print("=========================================\n")

        print(f'Currently processing:')
    
    # Iterate through all cuts
    for dir_name in os.listdir(image_ot_path):
        if not dir_name.startswith("ot_matrix_"):
            continue
        
        # Get cut name
        cut_name = dir_name.replace("ot_matrix", "")

        # Progress
        if cfg.verbose:
            print(dir_name)

        # Get features of current cut
        cut_ot_path = Path(image_ot_path / dir_name)
        cut_feature_droplets_file_path = Path(image_feature_path / f'droplets{cut_name}.csv')
        cut_feature_droplets_df = pd.read_csv(cut_feature_droplets_file_path)

        # Compute and store ot matrices for current cut
        compute_and_store_results_cut(cfg, cut_name, cut_ot_path, image_results_path, cut_feature_droplets_df)