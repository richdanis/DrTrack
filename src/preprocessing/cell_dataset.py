import numpy as np
import pandas as pd
from skimage.transform import resize
import tqdm
import argparse

DATA_PATH = '/cluster/scratch/rdanis/data/'

def create_mapping_df(patches_df, tracking_df, regions=None, distance=10):
    """
    Creates a DataFrame mapping droplet patches across frames.
    
    The DataFrame contains only correctly tracked droplets. Each column represents a frame, 
    named 'patchN', where N is the frame number and the values are the droplet patches in that frame.
    
    Parameters:
    patches_df (DataFrame): DataFrame containing droplet patches.
    tracking_df (DataFrame): DataFrame containing tracking information.
    regions (optional): Specific regions to consider.

    Returns:
    DataFrame: A DataFrame with droplet patches mapped across frames.
    """
    mapping = None
    columns = ["dropletIdNext", "center_row", "center_col"]
    num_frames = len(patches_df[0])
    
    # use tqdm to label the progress bar
    for i in tqdm.tqdm(range(num_frames-1), desc="Creating DataFrame..."):

        # get frames
        frame2 = pd.DataFrame(patches_df[0].iloc[i+1])

        # drop nan values and rows with no cells
        frame2 = frame2.dropna()
        frame2 = frame2[frame2["nr_cells"] > 0]
        frame2 = frame2.rename(columns={'patch': 'patch' + str(i+1)})

        if mapping is None:
            
            mapping = tracking_df[tracking_df["framePrev"] == i]

            # drop nan values and rows with no cells
            frame1 = pd.DataFrame(patches_df[0].iloc[i])
            frame1 = frame1.dropna()
            frame1 = frame1[frame1["nr_cells"] > 0]

            frame1 = frame1.rename(columns={'patch': 'patch' + str(i)})
            columns.append('patch' + str(i))

            mapping = mapping.merge(frame1, how="inner", left_on="dropletIdPrev", right_on="droplet_id")
        
        else:
            # join next frame
            mapping = mapping.merge(tracking_df[tracking_df["framePrev"] == i], \
                                    how="inner", left_on="dropletIdNext", \
                                    right_on="dropletIdPrev",  suffixes=("_x", None))

        columns.append('patch' + str(i+1))

        mapping = mapping.merge(frame2, how="inner", \
                                left_on="dropletIdNext", \
                                right_on="droplet_id", \
                                suffixes=('_x',None))

        mapping = mapping.dropna()

        # drop all rows where abs(center_x - center > 10)
        mapping = mapping[abs(mapping["center_row_x"] - mapping["center_row"]) < distance]
        mapping = mapping[abs(mapping["center_col_x"] - mapping["center_col"]) < distance]

        mapping = mapping[columns]
    
    columns = columns[3:]

    return mapping[columns]


def turn_to_numpy(df):
    """
    Converts a DataFrame into a numpy array.
    
    Each column of the DataFrame is resized and concatenated to form the final numpy array.
    
    Parameters:
    df (DataFrame): The DataFrame to be converted.

    Returns:
    numpy array: The resulting numpy array after conversion.
    """
    
    dataset = None

    for i in tqdm.tqdm(range(len(df.columns)), desc="Turning to numpy array..."):

        if i == 0:
            dataset = resize_patch_column(df[df.columns[i]])

        else:
            dataset = np.concatenate((dataset, resize_patch_column(df[df.columns[i]])), axis=1)

    return dataset


def resize_patch_column(df):
    """
    Resizes all patches in the DataFrame to 40x40.
    
    Each patch in the DataFrame is resized to a 2D 40x40 patch and stored in a numpy array.
    
    Parameters:
    df (DataFrame): The DataFrame containing patches to be resized.

    Returns:
    numpy array: The resulting numpy array containing resized patches.
    """

    resized = np.empty((len(df),1, 2, 40, 40))
    
    for i in range(len(df)):

        patch = df.iloc[i]
        patch = resize(patch, (2,40,40))
        patch = np.expand_dims(patch, axis=0)

        resized[i] = patch

    return resized


def main():
    """
    Main function to load patches, create a mapping, convert it to a numpy array, and save it.
    
    The function takes command line arguments for the name of the mapping file and the regions to consider in the image.
    It loads patches from a file, creates a mapping using the create_mapping_df function, converts the mapping to a numpy array,
    and saves the resulting numpy array.
    
    Command Line Arguments:
    --fname: Name of the input nd2 file (without .nd2 ending).
    --regions: Regions to consider in image. Format: x1 y1 x2 y2 ... Default is None.
    --distance: Maximum distance between droplets to be considered a match between two frames. Default is 10.
    """
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--fname', type=str, help='Name of the input image. (without .nd2 ending)')
    parser.add_argument('--regions', nargs='+', default=None, help='Regions to consider in image. Format: x1 y1 x2 y2 ...')
    parser.add_argument('--distance', type=int, default=10, help='Maximum distance between droplets to be considered a match between two frames.')

    args = parser.parse_args()

    fname = args.fname
    regions = args.regions
    distance = args.distance

    # make fname lower case and replace spaces with underscores
    fname = fname.lower()
    fname = fname.replace(" ", "_")

    # load patches
    patches = np.load(DATA_PATH + '03_features/fulldataset_' + fname + '.npy', allow_pickle=True)
    patches_df = pd.DataFrame(patches)

    # read tracking csv into dataframe
    tracking_df = pd.read_csv(DATA_PATH + '05_results/tracking_' + fname + '.csv')

    # create the chains of droplet patches
    mapping = create_mapping_df(patches_df, tracking_df, regions, distance)

    # turn to numpy array
    mapping = turn_to_numpy(mapping)

    # print total amount of patches
    print("Shape of dataset: " + str(mapping.shape))

    print("Saving mapping...")
    np.save(DATA_PATH + 'cell_datasets/' + fname, mapping)

if __name__ == '__main__':
    main()
