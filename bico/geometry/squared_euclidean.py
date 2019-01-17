import numpy as np

from nearpy.distances.distance import Distance


def squared_euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = x - y
    return np.sum(d ** 2)


class SquaredEuclideanDistance(Distance):
    """ Squared Euclidean distance for nearpy data structures """

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes squared euclidean distance between vectors x and y. Returns float.
        """
        return squared_euclidean_distance(x, y)
