import cv2 as cv
import numpy as np

# Local imports
from .nms import nms, canny_nms


def masked_dilate(image: np.array, 
                  mask: np.array, 
                  kernel: np.array) -> np.array:
    """
    Dilates an image while taking a mask into account.

    Parameters
    ----------
    image : np.ndarray
        The input image as a numpy ndarray.
    mask : np.ndarray
        The mask as a numpy ndarray.
    kernel : np.ndarray
        The kernel as a numpy ndarray.

    Returns
    -------
    np.ndarray
        The dilated image as a numpy ndarray.
    """
    return cv.morphologyEx(image, cv.MORPH_DILATE, kernel, iterations = 1) * mask


def cell_finding(cfg, droplet_circles: list, 
                 raw_dapi: np.ndarray, 
                 raw_bf: np.ndarray) -> np.ndarray:
    """
    Finding the cells in the image -- IMPORTANT not the droplets
    ----------
    Parameters:
    droplet_circles: list
        list of droplets which were detected
    raw_dapi: np.ndarray
        raw dapi image
    raw_bf: np.ndarray
        raw bf image
    ----------
    Returns:
    ans: 4d numpy array 
        with the following axes: Frames, Channels, Y (rows) and X (cols).
    """

    disable_tqdm = True 
    if cfg.verbose == True:
        disable_tqdm = False

    # memorize shape of whole image
    s = raw_dapi.shape

    # an auxiliary all-ones kernel which is used later
    kernel = np.ones((3, 3))

    # Return tensor. The tensor consists of 3 matrices of the same size. 
    # Matrix 1: mask that shows where significant peaks are detected
    # Matrix 2: stores intensity values of peaks  
    # Matrix 3: stores persistence values of peaks.
    ans = np.zeros((3, s[0], s[1]), dtype = np.float32)


    # We iterate over every droplet and aim to find the cells in that droplet
    for i in droplet_circles:

        # Get the center and radius of the droplet
        center = np.asarray((i[0], i[1]), dtype = np.int32)
        radius = i[2]

        # Define the the size of the window of interest we want to focus on for this droplet
        window_dim = radius + 10

        # Compute the effective size of the window of interest while taking into account the borders of the image
        window_rows = np.asarray((max(0, center[0] - window_dim), min(s[0], center[0] + window_dim + 1)), dtype = np.int32)
        window_cols = np.asarray((max(0, center[1] - window_dim), min(s[1], center[1] + window_dim + 1)), dtype = np.int32)

        # Compute the coordinates in the image which span the window of interest we want to take a closer look at (takes into account the image boundaies)
        target_rows = window_rows - (center[0] - window_dim)
        target_cols = window_cols - (center[1] - window_dim)
        # patch = np.zeros((2 * window_dim + 1, 2 * window_dim + 1), dtype = np.uint16)

        # Create a temporary local mask which will suppress everything outside the droplet 
        local_mask = np.zeros((2 * window_dim + 1, 2 * window_dim + 1), dtype = np.float32)
        cv.circle(local_mask, np.asarray([window_dim, window_dim]), radius, 1.0, -1)

        # Cut out the raw dapi-patch in the region of interest with the droplet in the center
        raw_patch = np.zeros((2 * window_dim + 1, 2 * window_dim + 1), dtype = np.float32)
        raw_patch[target_rows[0]: target_rows[1], target_cols[0]: target_cols[1]] = raw_dapi[window_rows[0]: window_rows[1], window_cols[0]: window_cols[1]] * 2.0**(-16)
        
        # Compute a mask which tells us, for the "raw_patch" that we have, where we have actual data from the image. 
        # This mask will be all ones if we are not close to an image boundary.
        # If we are close to an image boundary, this mask will be zero where we exceed the image boundary 
        signal_mask = np.zeros((2 * window_dim + 1, 2 * window_dim + 1), dtype = np.float32)
        signal_mask[target_rows[0]: target_rows[1], target_cols[0]: target_cols[1]] = 1.0

        # Cuts out the raw bf channel around the droplet.
        # This is actually not used at all for finding cells and is only used for debugging in order to visualize what we are doing.
        raw_bf_patch = np.zeros((2 * window_dim + 1, 2 * window_dim + 1), dtype = np.float32)
        raw_bf_patch[target_rows[0]: target_rows[1], target_cols[0]: target_cols[1]] = raw_bf[window_rows[0]: window_rows[1], window_cols[0]: window_cols[1]] * 2.0**(-16)
        
        # This handles boundary issues and sets the region of the cropped out patch which exceeds image boundaries, 
        # to the median of the brightness inside the droplet, which should be basically the brightness of the background.
        median_of_relevant_pixels = np.median(raw_patch[signal_mask * local_mask == 1.0])
        raw_patch[signal_mask == 0.0] = median_of_relevant_pixels
        
        # Need to normalize the image because the range of values actually used in the raw image is small compared to the possible range supported by the uint16 format.
        normalized_raw_patch = (raw_patch - raw_patch.min()) / (raw_patch.max() - raw_patch.min())

        # This median blurred image should capture the intensity bias in the different regions of the patch, without capturing the peaks in it
        flatfield_normalized_raw_base = np.float32(cv.medianBlur(np.uint8(normalized_raw_patch * 255), 2 * 5 + 1) / 255.0)

        # Subtract the intensity bias from the image and de-normalize
        corrected_raw_patch = raw_patch - (flatfield_normalized_raw_base * (raw_patch.max() - raw_patch.min()) + raw_patch.min())
        corrected_raw_patch = corrected_raw_patch - corrected_raw_patch.min()

        # Now that we have a raw patch without brightness bias, we slightly blur it to get rid of excessive noise
        corrected_patch = cv.GaussianBlur(corrected_raw_patch, (3, 3), 0)

        # All pixel intensities in the relvant part of the patch (ie that part that is inside the droplet and inside the image bundaries)
        relevant_pixels_list = corrected_patch[local_mask == 1.0]

        # Compute a threshold below which we believe there is only background. 
        background_threshold = np.quantile(relevant_pixels_list, 0.8)

        # The parts of the patch which we believe are guaranteed background
        guaranteed_background_mask = (corrected_patch <= background_threshold) * 1.0
        
        # Copy the background mask which we will use later when expanding this region of background
        guaranteed_background_mask_dilated = np.copy(guaranteed_background_mask)

        # Do non-maxima-suppression (ie finding all single pixels that stick out from the image in terms of brightness) and filter them to the region of interest
        patch_nms = nms(corrected_patch) * local_mask

        # Find all minima in the dapi channel, where a pixel counts as minimum if there exists at least one direction in which it is a minimum.
        # this basically means we find the valleys in the brightness-landscape
        inversepatch_nms = canny_nms(-corrected_raw_patch)

        # Now we will do 20 times a morphological operation.
        for i in range(20):
            # We grow the mask "guaranteed_background_mask_dilated" by growing the mask along the valleys found in "inversepatch_nms"
            # The heuristic here is that if a pixel in a valley is background, all other pixels in the same valley should also be background.
            guaranteed_background_mask_dilated = masked_dilate(guaranteed_background_mask_dilated, inversepatch_nms, kernel)
        
        # Add back the region of guaranteed backgrounds that we had earlier to this valley-grown mask
        guaranteed_background_mask_final = np.logical_or(guaranteed_background_mask, guaranteed_background_mask_dilated)
        # Do one closing morph op to get rid of noise and close up sme holes.
        guaranteed_background_mask_final = cv.morphologyEx(guaranteed_background_mask_final * 1.0, cv.MORPH_CLOSE, kernel, iterations = 1)

        # Now we will do 5 iterations in which we add all pixels to the background,
        # which do not significantly deviate from the current set of pixels we believe are background
        for i in range(5):
            # Get all pixels we believe are guaranteed background
            nullhypothesis = corrected_raw_patch[guaranteed_background_mask_final == 1.0]

            # Compute statistics of the background pixels
            hypo_mean = np.mean(nullhypothesis)
            hypo_std = np.std(nullhypothesis)

            # Mark as background all pixels that do not exceed a certain deviation from the background in terms of brightness
            guaranteed_background_mask_final[corrected_patch <= hypo_mean + 3 * hypo_std] = 1.0

        # Filter out all maxima in the dapi channel which are in the background
        peaks_nms = patch_nms * (1.0 - guaranteed_background_mask_final) 

        # This matrix stores the results which we will then copy over into "ans"        
        peaks = np.zeros((3, peaks_nms.shape[0], peaks_nms.shape[1]), dtype = np.float32)
        peaks[0, :, :] = peaks_nms
        peaks[1, :, :] = peaks_nms

        # Compute statistics of the background brightness in order to compute how bright a peak is compared to the background 
        noise_std = np.std(corrected_raw_patch[guaranteed_background_mask_final == 1.0])
        noise_mean = np.mean(corrected_raw_patch[guaranteed_background_mask_final == 1.0])

        # Compute the intensity score for the peaks
        peaks[1, :, :] = (corrected_patch - noise_mean) / (10.0 * noise_std) * peaks[0, :, :]

        # Where we found significant peaks
        peaks_detected_idxs = np.argwhere(peaks_nms != 0.0)

        # Compute a persistency score for every significant peak we found
        if peaks_detected_idxs.shape[0] > 0:

            # for every peak, find the 10 nearest pixels that are marked as background and compute the average distance from the
            # peak to those pixels. That is the persistency score.
            guaranteed_background_idxs = np.argwhere(guaranteed_background_mask_final == 1.0)
            distance_vecs = guaranteed_background_idxs[:, None, :] - peaks_detected_idxs[None, :, :]
            distances = np.linalg.norm(distance_vecs, axis = 2)
            distances.partition(10, axis = 0)
            mean_distances = np.mean(distances[: 10, :], axis = 0)

            presistency_score = mean_distances

            peaks[(np.repeat(2, mean_distances.size), peaks_detected_idxs[:, 0], peaks_detected_idxs[:, 1])] = presistency_score
        
        # Insert the local found cells and signals into the global frame
        ans[:, window_rows[0]: window_rows[1], window_cols[0]: window_cols[1]] = ans[:, window_rows[0]: window_rows[1], window_cols[0]: window_cols[1]] + peaks[:, target_rows[0]: target_rows[1], target_cols[0]: target_cols[1]]

    return ans
