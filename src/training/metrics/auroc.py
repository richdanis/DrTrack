from typing import Callable, List, Tuple

import numpy as np


def get_l2_distances(array1: np.array, array2: np.array) -> np.array:
    """Computes l2 distance between corresponding vectors in array1 and array2."""

    assert array1.shape == array2.shape;
    "To get distances between corresponding embeddings, arrays should be of the same shape."

    return np.linalg.norm(array2 - array1, axis=1)


def get_cosine_distances(array1: np.array, array2: np.array) -> np.array:
    """Computes cosine distance between corresponding vectors in array1 and array2."""

    assert array1.shape == array2.shape;
    "To get distances between corresponding embeddings, arrays should be of the same shape."

    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.sum(array1 * array2, axis=1)
    magnitude1 = np.linalg.norm(array1, axis=1)
    magnitude2 = np.linalg.norm(array2, axis=1)

    # Compute the cosine similarity
    cosine_similarity = dot_product / (magnitude1 * magnitude2)

    # Compute the cosine distance
    cosine_distance = 1 - cosine_similarity

    return cosine_distance


def calculate_tprs_fprs(array1: np.array, array2: np.array) -> Tuple[np.array, np.array]:
    """Computes tprs and fprs by thresholding on smallest k elements of joint `array1` and `array2`.

    Returned tprs and fprs are nonincreasing by construction.
    """
    # Combine the elements from array1 and array2, and create labels (1 for array1, 0 for array2)
    combined = np.concatenate([array1, array2])
    labels = np.concatenate([np.ones(len(array1)), np.zeros(len(array2))])

    # Sort the combined data in descending order
    sorted_indices = np.argsort(combined)
    sorted_labels = labels[sorted_indices]

    # Initialize variables for ROC curve
    true_positive_rates = []
    false_positive_rates = []
    num_positives = np.sum(sorted_labels)
    num_negatives = len(sorted_labels) - num_positives

    # Iterate through different values of k (threshold)
    for k in range(1, len(sorted_labels) + 1):
        true_positives = np.sum(sorted_labels[:k])
        false_positives = k - true_positives
        true_positive_rate = true_positives / num_positives
        false_positive_rate = false_positives / num_negatives

        true_positive_rates.append(true_positive_rate)
        false_positive_rates.append(false_positive_rate)

    return np.array(true_positive_rates), np.array(false_positive_rates)


def remove_duplicated_fprs(tprs, sorted_fprs) -> Tuple[List, List]:
    mask = np.concatenate([[True], sorted_fprs[1:] != sorted_fprs[:-1]])
    return tprs[mask], sorted_fprs[mask]


def calculate_auroc(array1: np.array, array2: np.array, negative_sample_mode: str = 'random', dist: str = 'l2') -> float:
    if dist == 'l2':
        dist_fun = get_l2_distances
    elif dist == 'cosine':
        dist_fun = get_cosine_distances
    else:
        raise NotImplementedError(f'Distance {dist} not implemented')

    positive_dist = dist_fun(array1, array2)

    if negative_sample_mode == 'rev':
        array2 = array2[::-1]
    elif negative_sample_mode == 'roll':
        array2 = np.roll(array2, shift=1)
    elif negative_sample_mode == 'random':
        array2 = np.random.permutation(array2)
    else:
        raise NotImplementedError(f'Negative sample mode {negative_sample_mode} not implemented')

    negative_dist = dist_fun(array1, array2)

    tprs, fprs = calculate_tprs_fprs(positive_dist, negative_dist)
    # fprs are sorted increasingly by construction
    tprs, fprs = remove_duplicated_fprs(tprs, fprs)

    roc = np.interp(np.linspace(0, 1, 100), fprs, tprs)
    auroc = np.mean(roc)

    return auroc
