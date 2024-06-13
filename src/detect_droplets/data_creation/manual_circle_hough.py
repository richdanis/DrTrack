# Computer vision for circle detection
import cv2 as cv

# Handling arrays
import numpy as np

# Local imports
from . import find_hough_circle, nms

def f32_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Transforms float32 images to uint8 images. Assumes the range of the input image is correct.

    Parameters
    ----------
    img : np.ndarray
        The input image as a numpy ndarray.
    
    Returns
    -------
    np.ndarray
        The input image as a numpy ndarray with datatype uint8.
    """
    return np.uint8(img * 255)


def uint8_to_f32(img: np.ndarray):
    """
    Transforms uint8 images to float32 images. Assumes the range of the input image is correct.

    Parameters
    ----------
    img : np.ndarray
        The input image as a numpy ndarray.
    
    Returns
    -------
    np.ndarray
        The input image as a numpy ndarray with datatype float32.
    """
    return np.float32(img / 255.0)


def manual_circle_hough(img: np.ndarray, 
                        refine: bool, 
                        bf_is_inverted: bool = False, 
                        noise_level_param: float = 0.3, 
                        radius_min: int = 12, 
                        radius_max: int = 25):
    """
    Detects circles in an image using the Hough transform.
    
    Parameters
    ----------
    img : np.ndarray
        The input image as a uint16 numpy ndarray (the greyscale base image).
    refine : bool
        A boolean flag indicating whether to refine the circles.
    bf_is_inverted : bool, optional
        A boolean flag indicating whether the brightfield channel is inverted. Default is False.
    noise_level_param : float, optional
        The noise level parameter. Default is 0.3.
    radius_min : int, optional
        The minimum radius for circle detection. Default is 12.
    radius_max : int, optional
        The maximum radius for circle detection. Default is 25.
    
    Returns
    -------
    np.ndarray
        Detected circles in the image.
    """
    
    # For the BF channel, bottom 80% of pixels is mostly background. Perform denoising.
    noise_level = noise_level_param

    img_denoised = (img - img.min()) / (img.max() - img.min())
    if not bf_is_inverted:
        img_denoised = 1.0 - img_denoised
    img_denoised = np.clip(img_denoised - np.quantile(img_denoised, noise_level), 0.0, 1.0)
    img_denoised = (img_denoised - img_denoised.min()) / (img_denoised.max() - img_denoised.min())

    # Depending on option, do refinement or not.
    # Refinement is done via the RANSAC algorithm.
    detected_circles = []
    if (not refine):
        img_denoised = cv.GaussianBlur(img_denoised, (3, 3), 0)
        # Works well for LM1, LM2, LM3, LM4, SM1, SM2, SM3
        preliminary_hough = np.uint16(np.around(cv.HoughCircles(f32_to_uint8(img_denoised), cv.HOUGH_GRADIENT, 1, 26, param1=80, param2=20, minRadius=radius_min, maxRadius=radius_max))) 
        # swaping the x and y coorinates and gettng the radius = i[2]
        detected_circles = [((i[1], i[0]), i[2]) for i in preliminary_hough[0, :]]

    else:
        img_denoised = cv.GaussianBlur(img_denoised, (3, 3), 0)
        img_edged = nms.canny_nms(img_denoised)
        # Works well for LM1, LM2, LM3, LM4, SM1, SM2, SM3
        preliminary_hough = np.uint16(np.around(cv.HoughCircles(f32_to_uint8(img_denoised), cv.HOUGH_GRADIENT, 1, 26, param1=80, param2=20, minRadius=radius_min, maxRadius=radius_max))) 
        preliminary_mask = np.zeros(img_denoised.shape, dtype = np.float32)
        for i in preliminary_hough[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv.circle(preliminary_mask, center, radius, 1.0, -1)

        # Create a mask that can suppress all detected circles
        preliminary_mask_negative = 1.0 - preliminary_mask
        erosion_kernel = np.ones((3, 3), dtype = np.float32)
        for i in preliminary_hough[0, :]:
            # Refine each circle
            center = (i[1], i[0])
            radius = i[2]
            # Some indexing madness to extract the relevant patch
            patch_x = (max(int(center[0]) - radius - 5, 0), min(int(center[0]) + radius + 5, img_denoised.shape[0] - 1))
            patch_y = (max(int(center[1]) - radius - 5, 0), min(int(center[1]) + radius + 5, img_denoised.shape[1] - 1))
            patch_keypoints = img_edged[patch_x[0]: patch_x[1], patch_y[0]: patch_y[1]]
            patch_edges = img_edged[patch_x[0]: patch_x[1], patch_y[0]: patch_y[1]]
            patch_mask = preliminary_mask_negative[patch_x[0]: patch_x[1], patch_y[0]: patch_y[1]]
            # Compute where the circle is estimated to be in the new patch
            center_in_patch = center - np.asarray([max(int(center[0]) - radius - 5, 0), max(int(center[1]) - radius - 5, 0)])
            # This operation results in patch_mask being a mask that suppresses all other circles except for the current one
            cv.circle(patch_mask, np.flip(center_in_patch) , radius, 1.0, -1)
            patch_mask = cv.morphologyEx(patch_mask, cv.MORPH_ERODE, erosion_kernel, iterations = 1)
            patch_mask = patch_mask * 0.9 + 0.1
            # This operation makes sure that we also ignore the center region of where we think the circle is in order to avoid noise from the beads
            cv.circle(patch_mask, np.flip(center_in_patch) , 10, 0.0, -1)
            patch_keypoints = patch_keypoints * patch_mask
            # Get the refined circle estimate
            refined_circle = find_hough_circle.circle_RANSAC3(patch_keypoints, patch_edges, 15, 35)
            if refined_circle is not None:
                refined_circle = (refined_circle[0] + max(int(center[0]) - radius - 5, 0), refined_circle[1] + max(int(center[1]) - radius - 5, 0), int(refined_circle[2]))
                center = (refined_circle[0], refined_circle[1])
                radius = refined_circle[2]
                detected_circles.append((refined_circle[0], refined_circle[1], radius))

    return detected_circles
