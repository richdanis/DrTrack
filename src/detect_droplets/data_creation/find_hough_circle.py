import cv2 as cv
import numpy as np

def circle_RANSAC3(points_img: np.ndarray, 
                   edge_img: int, 
                   rmin: int, 
                   rmax: int) -> set:
    """
    RANSAC Circle Fitting Function
    This function takes a binary image containing points on or near a circular shape and fits a circle using the RANSAC algorithm. 
    RANSAC is used to robustly estimate the circle's center and radius by randomly sampling and selecting the best circle that fits the data points.
    ----------
    Parameters:
    Inputs:
    points_img: np.ndarray
        Binary image with points on or near the circle.
    edge_img: int
        not used here, can be ignored - Edge image used for filtering connected components. 
    rmin: int
        not used here, can be ignored - Minimum radius of the circle to be detected.
    rmax: int
        not used here, can be ignored - Maximum radius of the circle to be detected.
    ----------
    Returns:
    Circle parameters: (int, int, int)
        A tuple containing the estimated circle's center coordinates (x, y) and its radius.
    ----------
    Note: 
    The function may return `None` if no circle is found within the given criteria.
    """
    np.random.seed(11000)
    s = points_img.shape

    tmp1 = np.uint8(points_img * 255)
    components, labels = cv.connectedComponents(tmp1, connectivity = 8, ltype = cv.CV_32S)
    cc_mask = np.ones(s)
    for i in range(components):
        if np.sum(labels == i) < 20:
            cc_mask[labels == i] = 0.0
    filtered_points_img = points_img * cc_mask

    nnz = np.sum(filtered_points_img == 1.0)
    if nnz < 3:
        return None

    coords = np.transpose(np.asarray(np.where(filtered_points_img == 1.0)))

    nr_of_samples = 500
    random_idxs = np.random.randint(0, nnz, size = (3, nr_of_samples), dtype = int)
    random_idxs[1, :] = np.random.randint(0, nnz - 1, size = nr_of_samples, dtype = int)
    random_idxs[2, :] = np.random.randint(0, nnz - 2, size = nr_of_samples, dtype = int)

    random_idxs[1, :] = random_idxs[1, :] + 1 * (random_idxs[1, :] >= random_idxs[0, :])
    random_idxs[2, :] = random_idxs[2, :] + 1 * (random_idxs[2, :] >= np.min(random_idxs[0: 2, :], axis = 0)) 
    random_idxs[2, :] = random_idxs[2, :] + 1 * (random_idxs[2, :] >= np.max(random_idxs[0: 2, :], axis = 0))

    coords1 = coords[random_idxs[0, :]]
    coords2 = coords[random_idxs[1, :]]
    coords3 = coords[random_idxs[2, :]]

    midpoints1 = (coords1 + coords2) / 2.0
    midpoints2 = (coords2 + coords3) / 2.0

    line_directions1 = coords1 - coords2
    line_directions1 = np.transpose(np.asarray([line_directions1[:, 1], -line_directions1[:, 0]]))

    line_directions1 = line_directions1 / np.linalg.norm(line_directions1, axis = 1)[:, None]
    line_directions2 = coords2 - coords3
    line_directions2 = np.transpose(np.asarray([line_directions2[:, 1], -line_directions2[:, 0]]))
    line_directions2 = line_directions2 / np.linalg.norm(line_directions2, axis = 1)[:, None]

    sample_viability = (np.abs(np.sum(line_directions1 * line_directions2, axis = 1)) <= 0.8)
    if np.sum(sample_viability) == 0:
        return None

    coords1 = coords1[sample_viability, :]
    coords2 = coords2[sample_viability, :]
    coords3 = coords3[sample_viability, :]
    midpoints1 = midpoints1[sample_viability, :]
    midpoints2 = midpoints2[sample_viability, :]
    line_directions1 = line_directions1[sample_viability, :]
    line_directions2 = line_directions2[sample_viability, :]

    cs = midpoints1 - midpoints2

    As = np.transpose(np.asarray([-np.transpose(line_directions1), np.transpose(line_directions2)]), (2, 1, 0))

    results = np.linalg.solve(As, cs)

    centers = (midpoints1 + line_directions1 * results[:, 0, None] + midpoints2 + line_directions2 * results[:, 1, None]) / 2.0
    radii = (np.linalg.norm(centers - coords1, axis = 1) + np.linalg.norm(centers - coords2, axis = 1) + np.linalg.norm(centers - coords3, axis = 1)) / 3.0

    distances_to_centers = np.transpose(np.linalg.norm(centers - coords[:, None, :], axis = 2))

    distance_within_rad = np.logical_and(distances_to_centers <= radii[:, None] + 1.0, distances_to_centers >= radii[:, None] - 1.0)
    score = np.sum(distance_within_rad, axis = 1)

    winner = np.argmax(score)
    center_to_return = centers[winner, :]
    radius_to_return = round(radii[winner])

    return (round(center_to_return[0]), round(center_to_return[1]), radius_to_return)
