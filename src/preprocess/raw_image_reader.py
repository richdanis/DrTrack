from typing import Optional

import numpy as np
import nd2

def get_image_as_ndarray(channels: list,
                         path_to_image: str,
                         all_frames: bool = True,
                         all_channels: bool = False,
                         frames: Optional[list] = None,
                         pixel: int = -1) -> np.ndarray:
    """
    Get the .nd2 image as a ndarray.
    ----------
    Parameters:
    channels: list[str]
        channels wanted from: [DAPI, FITC, TRITC, Cy5, BF] 
            'BF' is the identifier for the bright-field images,
            'DAPI' is the identifier for the DAPI channel
    path_to_image: str
        relative path to the clip (clip as .nd2 file)
    all_frames: bool (default = True)
        if true returns all frames - otherwise as indicated in frames
    frames: Optional[list] - a list of frames in case not all of them are used
    all_channels: bool (default = False)
        if true returns all channels - otherwise as indicated in channels
      pixels: int
        num of pixels to cut out of the full image, if -1 the full image is taken
    ----------
    Returns:
    ndarray: 4d numpy array (uint16)
        with the following axes: Frames, Channels, Y (rows) and X (cols).
    """

    f = nd2.ND2File(path_to_image)

    nr_frames = f.sizes['T']
    nr_channels = f.sizes['C']
    nr_rows = f.sizes['Y']
    nr_cols = f.sizes['X']

    print(f'Nr frames: {nr_frames}')
    print(f'Nr channels: {nr_channels}')
    print(f'Image dimensions: {nr_rows}x{nr_cols}')

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

    if all_frames:
        frames = range(nr_frames)
    else:
        frames = frames

    fullimage = f.asarray()

    if pixel == -1:
        output = (fullimage[frames, :, :, :])[:, channel_idx_precompute, :, :]
    else:
        output = (fullimage[frames, :, :pixel, :pixel])[:, channel_idx_precompute, :, :]

    f.close()
    return output

def get_image_cut_as_ndarray(cfg,
                            channels: list,
                            path_to_image: str,
                            upper_left_corner: tuple,
                            pixel_dimensions: tuple,
                            all_frames: bool = True, 
                            all_channels: bool = False, 
                            frames = None,
                            pixel: int = -1) -> np.ndarray:
    """
    Get the clip as a ndarray - PRIMARY FUNCTION
    ----------
    Parameters:″
    frames: list[int]
        frames wanted from the clip
    channels: list[str]
        channels wanted from: [DAPI, FITC, TRITC, Cy5, BF] 
            'BF' is the identifier for the bright-field images,
            'DAPI' is the identifier for the DAPI channel
    path_to_image: str
        relative path to the clip (clip as .nd2 file)
    all_frames: bool (default = True)
        if true returns all frames - otherwise as indicated in frames 
    all_channels: bool (default = False)
        if true returns all channels - otherwise as indicated in channels 
    ----------
    Returns:
    ndarray: 4d numpy array (uint16)
        with the following axes: Frames, Channels, Y (rows) and X (cols).
    """

    f = nd2.ND2File(path_to_image)

    nr_frames = f.sizes['T']
    nr_channels = f.sizes['C']
    nr_rows = f.sizes['Y']
    nr_cols = f.sizes['X']

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

    if all_frames:
        frames = range(nr_frames)
    else:
        frames = frames
    
    fullimage = f.asarray()

    # Ensure that the lower right corner is included in the image
    y_upper_left, x_upper_left = upper_left_corner
    size_y, size_x = pixel_dimensions
    y_end = y_upper_left + size_y
    if y_upper_left + 2*size_y > nr_rows:
        y_end = nr_rows
    x_end = x_upper_left + size_x
    if x_upper_left + 2*size_x > nr_cols:
        x_end = nr_cols

    if pixel == -1:
        output = (fullimage[frames, :, y_upper_left:y_end, x_upper_left:x_end])[:, channel_idx_precompute, :, :]
    else:
        output = (fullimage[frames, :, :pixel, :pixel])[:, channel_idx_precompute, :, :]
    
    f.close()
    
    return output