import numpy as np
from bico.geometry.squared_euclidean import SquaredEuclideanDistance
from bico.nearest_neighbor.base import NearestNeighbor, NearestNeighborResult
from nearpy import Engine
from nearpy.filters import DistanceThresholdFilter
from nearpy.hashes import RandomBinaryProjections


class RandomBinaryNN(NearestNeighbor):
    """ Nearest neighbor implementation by using random binary trees from nearpy package """

    def __init__(self, dimension: int, number_projections: int, threshold: float):
        """
        :param dimension:
            Number of dimensions of input points
        :param number_projections:
            Number of random projections used for finding nearest neighbors.
            Trade-off: More projections result in a smaller number of false positives in candidate set
        :param threshold:
            Distance threshold for definition nearest: all points within this specific distance
        """
        self.rbp = RandomBinaryProjections('rbp', number_projections)
        self.sqdist = SquaredEuclideanDistance()
        self.ann_engine = Engine(dimension, lshashes=[self.rbp], distance=self.sqdist,
                                 vector_filters=[DistanceThresholdFilter(threshold)])

    def insert_candidate(self, point: np.ndarray, metadata):
        self.ann_engine.store_vector(point, data=metadata)

    def get_candidates(self, point: np.ndarray):
        return [NearestNeighborResult(res[0], res[1], res[2])
                for res in self.ann_engine.neighbours(point)]
