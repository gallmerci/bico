import numpy as np
from bico.geometry.squared_euclidean import squared_euclidean_distance
from bico.nearest_neighbor.base import NearestNeighbor, NearestNeighborResult
from typing import List


class SimpleProjection(NearestNeighbor):
    """ Nearest neighbor implementation by projecting points into buckets using random dot products """
    def __init__(self, dimension: int, number_projections: int, threshold_filter: float):
        """
        :param dimension:
            Number of dimensions of input points
        :param number_projections:
            Number of random projections used for finding nearest neighbors.
            Trade-off: More projections result in a smaller number of false positives in candidate set
        :param threshold_filter:
            Distance threshold for definition nearest: all points within this specific distance
        """
        self.dimension = dimension
        self.number_projections = number_projections
        self.threshold_filter = threshold_filter
        self.__create_projections()

    def __create_projections(self):
        self.projections = np.array(list(np.random.standard_normal(self.dimension)
                                         for _ in range(self.number_projections)))
        self.buckets = [dict() for _ in range(self.number_projections)]

    def project(self, point: np.ndarray) -> np.ndarray:
        return self.projections.dot(point)

    def get_candidates(self, point: np.ndarray) -> List[NearestNeighborResult]:
        proj_values = self.project(point)
        bucket_values = self.get_bucket_values(proj_values)
        smallest_bucket = min([x[1].get(bucket_values[x[0]], [])
                               for x in enumerate(self.buckets)], key=len)
        distances = [squared_euclidean_distance(p[0], point) for p in smallest_bucket]
        res = [NearestNeighborResult(p[0][0], p[0][1], p[1]) for p in zip(smallest_bucket, distances)]
        return sorted(res, key=lambda x: x.distance)

    def get_bucket_values(self, proj_values: np.ndarray) -> np.ndarray:
        return (proj_values / (2 * self.threshold_filter)).astype(int)

    def insert_candidate(self, point: np.ndarray, metadata):
        proj_values = self.project(point)
        data_point = (point, metadata)
        bucket_values = self.get_bucket_values(proj_values)
        for i, bucket in enumerate(self.buckets):
            cand_list = bucket.get(bucket_values[i], [])
            if len(cand_list) > 0:
                cand_list.append(data_point)
            else:
                bucket[bucket_values[i]] = [data_point]

