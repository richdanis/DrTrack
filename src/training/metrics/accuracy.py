from typing import Union, List

import numpy as np
from scipy.spatial import distance


def find_closest_neighbors_l2(array1: np.array, array2: np.array, topk: int = 1) -> np.array:
    """For vectors in array1 finds topk closest neighbors inside array2 using L2 distance."""
    closest_neighbors = []

    for vector1 in array1:
        distances = np.linalg.norm(array2 - vector1, axis=1)
        closest_neighbors.append(np.argsort(distances)[:topk])

    return np.array(closest_neighbors)


def find_closest_neighbors_cosine(array1: np.array, array2: np.array, topk: int = 1) -> np.array:
    """For vectors in array1 finds topk closest neighbors inside array2 using cosine distance."""

    distances = distance.cdist(array1, array2, 'cosine')
    distances_sort = np.argsort(distances, axis=1)[:, :topk]
    return distances_sort


def calculate_accuracy(array1: np.array, array2: np.array, topk: Union[List, int], dist: str = 'l2') -> List[float]:
    if dist == 'l2':
        find_closest_neighbors = find_closest_neighbors_l2
    elif dist == 'cosine':
        find_closest_neighbors = find_closest_neighbors_cosine
    else:
        raise NotImplementedError(f'Distance {dist} not implemented')

    if not isinstance(topk, List):
        topk = List[topk]

    closest_neighbors = find_closest_neighbors(array1, array2,
                                               topk=max(topk))

    accuracy = []
    num_embeddings = array1.shape[0]

    for k in topk:
        matched = 0
        for i, i_closest_neighbors in enumerate(closest_neighbors):
            # print(dist, i, i_closest_neighbors[:k])
            matched += (i in i_closest_neighbors[:k])

        accuracy.append(matched / num_embeddings)

    return accuracy
