# Types
from typing import Optional, Tuple, List
from pathlib import Path

# Other imports
import numpy as np
import nd2


def get_image_cut_as_ndarray(cfg,
                             channels: List[str],
                             path_to_image: Path,
                             upper_left_corner: Tuple[int, int],
                             pixel_dimensions: Tuple[int, int],
                             all_frames: bool = True, 
                             all_channels: bool = False, 
                             frames: Optional[List[int]] = None,
                             pixel: int = -1) -> np.ndarray:
    """
    Get the image as a ndarray.
    ----------
    Parameters:
    channels: list[str]
        channels wanted from: [DAPI, FITC, TRITC, Cy5, BF] 
            'BF' is the identifier for the bright-field images,
            'DAPI' is the identifier for the DAPI channel
    path_to_image: Path
        relative path to the image (image as .nd2 file)
    upper_left_corner: Tuple[int, int],
        coordinates of the upper left corner of the cutout (y,x)
    pixel_dimensions: Tuple[int, int],
        dimensions of the cutout (y,x)
    all_frames: bool (default = True)
        if true returns all frames - otherwise as indicated in frames 
    all_channels: bool (default = False)
        if true returns all channels - otherwise as indicated in channels 
    frames: Optional[list[int]] 
        a list of frames in case not all of them are used
    pixel: int
        num of pixels to cut out of the full image, if -1 the full image is taken
    ----------
    Returns:
    ndarray: 4d numpy array (uint16)
        with the following axes: Frames, Channels, Y (rows) and X (cols).
    """
    # Open nd2 file and check sizes
    f = nd2.ND2File(path_to_image)

    nr_frames = f.sizes['T']
    nr_channels = f.sizes['C']
    nr_rows = f.sizes['Y']
    nr_cols = f.sizes['X']

    # Get desired channels
    channel_idx_lookup = {}
    if all_channels:
        channels = []

    for c in f.metadata.channels:
        channel_name = c.channel.name
        if all_channels:
            channels.append(channel_name)
        channel_idx = c.channel.index
        if channel_name in channels:
            channel_idx_lookup[channel_name] = channel_idx

    channel_idx_precompute = []
    for ch_name in channels:
        channel_idx_precompute.append(channel_idx_lookup[ch_name])

    # Define range of frames
    if all_frames:
        frames = range(nr_frames)
    else:
        frames = frames
    
    fullimage = f.asarray()

    # Ensure that the lower right corner is included in the image
    y_upper_left, x_upper_left = upper_left_corner
    size_y, size_x = pixel_dimensions
    if pixel_dimensions == (-1,-1):
        size_y = nr_rows
        size_x = nr_cols
    y_end = y_upper_left + size_y
    if y_upper_left + 2*size_y > nr_rows:
        y_end = nr_rows
    x_end = x_upper_left + size_x
    if x_upper_left + 2*size_x > nr_cols:
        x_end = nr_cols

    # Get the image cut, and only the channels and frames wanted
    if pixel == -1:
        output = (fullimage[frames, :, y_upper_left:y_end, x_upper_left:x_end])[:, channel_idx_precompute, :, :]
    else:
        output = (fullimage[frames, :, y_upper_left:min(y_upper_left+pixel,y_end), x_upper_left:min(x_upper_left+pixel,x_end)])[:, channel_idx_precompute, :, :]
    
    # Close nd2 file
    f.close()
    
    return output