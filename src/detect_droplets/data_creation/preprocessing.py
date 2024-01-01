# Computer vision, image processing and arrays
import numpy as np
import cv2 as cv
import nd2

# Progress
from tqdm.auto import tqdm

# skimage imports
from skimage.filters import rank
from skimage import morphology
from skimage.util import img_as_ubyte

def get_clip_as_ndarray(channels: list,
                        path_to_image: str,
                        allFrames: bool = True, 
                        allChannels: bool = False, 
                        frames = None,
                        pixel: int = -1) -> np.ndarray:
    """
    Get the clip as a ndarray.
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
    allFrames: bool (default = True)
        if true returns all frames - otherwise as indicated in frames 
    allChannels: bool (default = False)
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

    print(f'Nr frames: {nr_frames}')
    print(f'Nr channels: {nr_channels}')
    print(f'Image dimensions: {nr_rows}x{nr_cols}')

    channel_idx_lookup = {}
    if allChannels:
        channels = []

    for c in f.metadata.channels:
        channel_name = c.channel.name
        if allChannels:
            channels.append(channel_name)
        channel_idx = c.channel.index
        if channel_name in channels:
            channel_idx_lookup[channel_name] = channel_idx

    channel_idx_precompute = []
    for ch_name in channels:
        channel_idx_precompute.append(channel_idx_lookup[ch_name])

    if allFrames:
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

