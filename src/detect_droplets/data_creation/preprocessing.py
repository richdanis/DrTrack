import numpy as np
import cv2 as cv
import nd2
from tqdm.auto import tqdm

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


def preprocess_alt_franc(image_path: str, 
                         pixel: int = -1) -> np.ndarray:
    """
    Preprocess the BF channel of a clip 
    ----------
    Parameters:
    path_to_image: str
        relative path to the clip (clip as .nd2 file)
    ----------
    Returns:
    ndarray: 4d numpy array (uint16 so be careful)
        with the following axes: Frames, Channels ['BF', 'DAPI'], Y (rows) and X (cols),
        where the BF channel has been processed with quantile and locally histogram equalized
    """

    clip = get_clip_as_ndarray(['BF', 'DAPI'], image_path, allFrames=True, allChannels=False, pixel=pixel)

    # For each frame, preprocess channels inplace
    clip[:, 0, :, :] = np.uint16(2**16 - (np.int32(clip[:, 0, :, :]) + 1))

    for frame in tqdm(clip, desc='BF preprocessing'):

        # Brightfield preprocessing
        bf_chan = np.float64(frame[0, :, :])
        bf_chan_low = np.quantile(bf_chan, 0.1)
        bf_chan_high = np.quantile(bf_chan, 0.995)
        bf_chan = np.clip((bf_chan - bf_chan_low) / (bf_chan_high - bf_chan_low), 0.0, 1.0)

        pullback_min = bf_chan.min()
        pullback_max = bf_chan.max()
        bf_pullback = (bf_chan - pullback_min) / (pullback_max - pullback_min)

        bf_pullback = np.clip((bf_pullback - np.quantile(bf_pullback, 0.5)) / (1.0 - np.quantile(bf_pullback, 0.5)), 0.0, 1.0)

        equalized = rank.equalize(img_as_ubyte(bf_pullback), footprint=morphology.disk(10)) / 255.0

        bf_pullback = bf_pullback * equalized

        smoothed = cv.GaussianBlur(bf_pullback, (3, 3), 0)
        frame[0, :, :] = np.uint16(smoothed * (2**16 - 1))

    return clip


def preprocess_alt_featextr(image_path: str,
                            pixel: int = -1) -> np.ndarray:

    """
    Preprocess the BF channel of a clip 
    ----------
    Parameters:
    path_to_image: str
        relative path to the clip (clip as .nd2 file)
    ----------
    Returns:
    ndarray: 4d numpy array (uint16 so be careful)
        with the following axes: Frames, Channels ['BF', 'DAPI'], Y (rows) and X (cols),
        where the BF and DAPI channels have been processed with quantile and locally histogram equalized
    """

    clip = get_clip_as_ndarray(['BF', 'DAPI'], image_path, allFrames=True, allChannels=False, pixel=pixel)

    # For each frame, preprocess channels inplace
    clip[:, 0, :, :] = np.uint16(2**16 - (np.int32(clip[:, 0, :, :]) + 1))

    for frame in tqdm(clip, desc='BF and DAPI preprocessing'):

        # Brightfield preprocessing
        bf_chan = np.float64(frame[0, :, :])
        bf_chan_low = np.quantile(bf_chan, 0.1)
        bf_chan_high = np.quantile(bf_chan, 0.995)
        bf_chan = np.clip((bf_chan - bf_chan_low) / (bf_chan_high - bf_chan_low), 0.0, 1.0)

        img_medianblurred = np.float64(cv.medianBlur(np.uint8(bf_chan * 255), 2 * 50 + 1) / 255.0)
        img_mediansharpened = np.clip(bf_chan - img_medianblurred, 0.0, 1.0)
        equalized_bf = rank.equalize(img_as_ubyte(img_mediansharpened), footprint=morphology.disk(10)) / 255.0
        img_mediansharpened = img_mediansharpened * equalized_bf
        thresh = np.quantile(img_mediansharpened, 0.5)
        img_mediansharpened[img_mediansharpened > thresh] = bf_chan[img_mediansharpened > thresh]

        frame[0] = cv.GaussianBlur(np.uint16(img_mediansharpened * (2**16 - 1)), (3, 3), 0)

        # DAPI preprocessing
        dapi_chan = np.float64(frame[1, :, :])
        dapi_chan_low = np.quantile(dapi_chan, 0.8)
        dapi_chan = np.clip((dapi_chan - dapi_chan_low) / ((2**16 - 1) - dapi_chan_low), 0.0, 1.0)
        img_medianblurred = np.float64(cv.medianBlur(np.uint8(dapi_chan * 255), 2 * 20 + 1) / 255.0)
        img_mediansharpened = np.clip(dapi_chan - img_medianblurred, 0.0, 1.0)
        equalized_dapi = rank.equalize(img_as_ubyte(img_mediansharpened), footprint=morphology.disk(10)) / 255.0
        img_mediansharpened = img_mediansharpened * equalized_dapi
        thresh = np.quantile(img_mediansharpened, 0.8)
        img_mediansharpened[img_mediansharpened > thresh] = dapi_chan[img_mediansharpened > thresh]

        frame[1] = np.uint16(img_mediansharpened * (2**16 - 1))

    clip[np.isnan(clip)] = 0.0
    return clip