import numpy as np
import pandas as pd
from skimage.transform import resize
import tqdm
import argparse
import os

DATA_PATH = '/cluster/scratch/rdanis/data/'
SAVE_PATH = '/cluster/scratch/rdanis/data/'


def create_mapping_df(patches_df, tracking_df, distance=10):
    """
    Creates a DataFrame mapping droplet patches across frames with position information.

    The DataFrame contains only correctly (this is an assumption) tracked droplets. Each column represents a frame, 
    named 'patchN', where N is the frame number and the values are the droplet patches in that frame.

    Parameters:
    patches_df (DataFrame): DataFrame containing droplet patches.
    tracking_df (DataFrame): DataFrame containing tracking information.

    Returns:
    DataFrame: A DataFrame with droplet patches mapped across frames with position information.
    """
    mapping = None
    columns = ["dropletIdNext"]
    num_frames = len(patches_df[0])

    # use tqdm to label the progress bar
    for i in tqdm.tqdm(range(num_frames-1), desc="Creating DataFrame..."):

        # get frames
        frame2 = pd.DataFrame(patches_df[0].iloc[i+1])

        # drop nan values and rows with no cells
        frame2 = frame2.dropna()
        frame2 = frame2[frame2["nr_cells"] > 0]
        frame2 = frame2.rename(columns={'patch': 'patch' + str(i+1),
                                        'center_row': 'x' + str(i+1),
                                        'center_col': 'y' + str(i+1)})

        if mapping is None:

            mapping = tracking_df[tracking_df["framePrev"] == i]

            # drop nan values and rows with no cells
            frame1 = pd.DataFrame(patches_df[0].iloc[i])
            frame1 = frame1.dropna()
            frame1 = frame1[frame1["nr_cells"] > 0]

            frame1 = frame1.rename(columns={'patch': 'patch' + str(i),
                                            'center_row': 'x' + str(i),
                                            'center_col': 'y' + str(i)})
            columns.append('patch' + str(i))
            columns.append('x' + str(i))
            columns.append('y' + str(i))

            mapping = mapping.merge(
                frame1, how="inner", left_on="dropletIdPrev", right_on="droplet_id")

        else:
            # join next frame
            mapping = mapping.merge(tracking_df[tracking_df["framePrev"] == i],
                                    how="inner", left_on="dropletIdNext",
                                    right_on="dropletIdPrev",  suffixes=("_x", None))

        columns.append('patch' + str(i+1))
        columns.append('x' + str(i+1))
        columns.append('y' + str(i+1))

        mapping = mapping.merge(frame2, how="inner",
                                left_on="dropletIdNext",
                                right_on="droplet_id")

        mapping = mapping.dropna()

        # drop all rows where abs(center_x - center > 10)
        mapping = mapping[abs(mapping["x" + str(i)] -
                              mapping["x" + str(i+1)]) < distance]
        mapping = mapping[abs(mapping["y" + str(i)] -
                              mapping["y" + str(i+1)]) < distance]

        mapping = mapping[columns]

    columns = columns[1:]

    return mapping[columns]


def local_negatives(mapped_patches, num_frames):
    """
    Samples negatives for each droplet in the dataset. Negatives are from the following frame. These are used then in the contrastive loss.

    Parameters:
    mapped_patches (DataFrame): DataFrame containing droplet patches mapped across frames with position information.
    num_frames (int): Number of frames in the dataset.

    Returns:
    numpy array: A numpy array containing the indices of the negatives for each droplet in the dataset.
    """
    # create empty index list
    indices = np.empty(
        (len(mapped_patches) * (num_frames - 1), 128), dtype=np.int16)

    # iterate over frames:
    for i in tqdm.tqdm(range(num_frames-1), desc="Creating local negatives..."):
        # iterate over droplets in frame i
        next_x = np.array(mapped_patches["x" + str(i+1)])
        next_y = np.array(mapped_patches["y" + str(i+1)])
        for j in range(len(mapped_patches)):
            # get positions
            curr_x = np.array(mapped_patches["x" + str(i)].iloc[j])
            curr_y = np.array(mapped_patches["y" + str(i)].iloc[j])

            # exclude j-th entry from next
            next_x_temp = np.delete(next_x, j)
            next_y_temp = np.delete(next_y, j)

            # calculate distances
            distances = np.sqrt((curr_x - next_x_temp) **
                                2 + (curr_y - next_y_temp)**2)

            # get 128 closest droplets
            closest = np.argsort(distances)[:128]

            # add length of list to closest
            closest = closest + len(mapped_patches) * (i + 1)

            # update indices
            indices[j + (i * len(mapped_patches)), :] = closest

    return indices


def get_labels(mapped_patches, num_frames):

    labels = np.empty(
        (len(mapped_patches) * (num_frames - 1),), dtype=np.int16)

    for i in range(num_frames-1):
        for j in range(len(mapped_patches)):
            labels[j + (i * len(mapped_patches))] = j + \
                (i+1) * len(mapped_patches)

    return labels


def turn_into_right_format(mapped_patches, num_frames):
    """
    Turns the DataFrame containing droplet patches into the right format for training.

    The DataFrame is flattened into a numpy array and the labels are created. The negatives are also created.

    Parameters:
    mapped_patches (DataFrame): DataFrame containing droplet patches mapped across frames with position information.
    num_frames (int): Number of frames in the dataset.

    Returns:
    numpy array: A numpy array containing the patches.
    numpy array: A numpy array containing the labels.
    numpy array: A numpy array containing the indices of the negatives for each droplet in the dataset.
    """
    # get negative indices
    neg_indices = local_negatives(mapped_patches, num_frames)

    # flatten patches into numpy array
    dataset = turn_to_numpy(mapped_patches, num_frames)

    # get labels
    labels = get_labels(mapped_patches, num_frames)

    # apply same random shuffling to dataset and labels
    permutation = np.arange(len(dataset))
    permutation[:len(labels)] = np.random.permutation(len(labels))

    # apply permutation to dataset and labels
    dataset = dataset[permutation]
    labels = labels[permutation[:len(labels)]]
    neg_indices = neg_indices[permutation[:len(labels)]]

    for i in tqdm.tqdm(range(len(labels)), desc="Applying permutation to indices..."):
        labels[i] = np.where(permutation == labels[i])[0][0]
        for j in range(len(neg_indices[i])):
            neg_indices[i][j] = np.where(
                permutation == neg_indices[i][j])[0][0]

    return dataset, labels, neg_indices


def turn_to_numpy(df, num_frames):
    """
    Converts the DataFrame containing droplet patches to a numpy array.
    Droplet patches are flattened to a 2D 40x40 patch and stored in a numpy array.

    Parameters:
    df (DataFrame): The DataFrame containing patches to be converted.
    num_frames (int): Number of frames in the dataset.

    Returns:
    numpy array: The resulting numpy array containing patches.
    """
    dataset = np.empty((num_frames * len(df), 2, 40, 40), dtype=np.float32)

    for i in tqdm.tqdm(range(num_frames), desc="Flatten to numpy array..."):
        for j in range(len(df)):
            dataset[j + i * len(df)] = df["patch" + str(i)].iloc[j]

    return dataset


def resize_patch_columns(df, num_frames):
    """
    Resizes all patches in the DataFrame to 40x40.

    Each patch in the DataFrame is resized to a 2D 40x40 patch and stored in a numpy array.

    Parameters:
    df (DataFrame): The DataFrame containing patches to be resized.

    Returns:
    numpy array: The resulting numpy array containing resized patches.
    """
    for i in tqdm.tqdm(range(num_frames), desc="Resizing patches..."):
        for j in range(len(df)):
            patch = df["patch" + str(i)].iloc[j]
            patch = resize(patch, (2, 40, 40))
            # updated dataframe
            df.at[j, "patch" + str(i)] = patch

    return df


def main():
    """
    Main function to load patches, create a mapping, convert it to a numpy array, and save it.

    Command Line Arguments:
    --fname: Name of the input nd2 file (without .nd2 ending).
    --distance: Maximum distance between droplets to be considered a match between two frames. Default is 10.
    """
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--fname', type=str,
                        help='Name of the input image. (without .nd2 ending)')
    parser.add_argument('--distance', type=int, default=10,
                        help='Maximum distance between droplets to be considered a match between two frames.')

    args = parser.parse_args()

    fname = args.fname
    distance = args.distance

    # make fname lower case and replace spaces with underscores
    fname = fname.lower()
    fname = fname.replace(" ", "_")

    # load patches
    patches = np.load(DATA_PATH + '03_features/fulldataset_' +
                      fname + '.npy', allow_pickle=True)
    patches_df = pd.DataFrame(patches)

    # read tracking csv into dataframe
    tracking_df = pd.read_csv(
        DATA_PATH + '05_results/tracking_' + fname + '.csv')

    # number of frames
    num_frames = len(patches_df[0])

    # create the chains of droplet patches
    mapping = create_mapping_df(patches_df, tracking_df, distance)

    # resize patches
    mapping = resize_patch_columns(mapping, num_frames)

    # turn into the right format
    dataset, label, neg_indices = turn_into_right_format(mapping, num_frames)

    # print total amount of patches
    print("Shape of dataset: " + str(dataset.shape))
    print("Shape of labels: " + str(label.shape))
    print("Shape of negative indices: " + str(neg_indices.shape))

    # modify save path
    SAVE_PATH = SAVE_PATH + 'local_datasets/' + fname + '/'

    # make directories
    os.makedirs(SAVE_PATH, exist_ok=True)

    print("Saving files...")
    np.save(SAVE_PATH + 'patches', dataset)
    np.save(SAVE_PATH + 'labels', label)
    np.save(SAVE_PATH + 'negatives', neg_indices)


if __name__ == '__main__':
    main()
