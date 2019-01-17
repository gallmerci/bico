import logging
import numpy as np
from bico.geometry.point import Point
from bico.nearest_neighbor.base import NearestNeighbor
from bico.utils.ClusteringFeature import ClusteringFeature
from datetime import datetime
from typing import Callable, TextIO, List

logger = logging.getLogger(__name__)


class BICONode:
    def __init__(self, level: int, dim: int, proj: int, bico: 'BICO',
                 projection_func: Callable[[int, int, float], NearestNeighbor]):
        self.level = level
        self.dim = dim
        self.proj = proj
        self.point_to_biconode = []
        self.projection_func = projection_func
        self.nn_engine = projection_func(dim, proj, bico.get_radius(self.level))
        self.num_cfs = 0
        self.bico = bico
        self.cf = ClusteringFeature(Point(np.zeros(dim)), Point(np.zeros(dim)), 0, 0)

    def insert_point(self, point_cf: ClusteringFeature) -> int:
        if self.bico.verbose:
            logger.debug("Insert point: {}".format(point_cf))
        # check whether geometry fits into CF
        if self.level > 0:
            if self.cf.size == 0:
                self.cf += point_cf
                self.cf.ref = point_cf.ref
            else:
                test = self.cf + point_cf
                cost = test.kmeans_cost(self.cf.ref)
                if self.bico.verbose:
                    logger.debug("Cost: " + str(cost) + ", Thresh: " + str(self.bico.get_threshold(self.level)))
                if cost < self.bico.get_threshold(self.level):
                    self.cf = test
                    return 0

        # search nearest neighbor and insert geometry there or open new BICONode
        candidates = []
        if self.num_cfs > 0:
            if self.bico.track_time:
                tstart = datetime.now()
            candidates = self.nn_engine.get_candidates(point_cf.ref.p)
            # candidates = self.ann_engine.neighbours(point_cf.ref.p)
            if self.bico.track_time:
                tend = datetime.now()
                if len(self.bico.time) < self.level + 1:
                    self.bico.time.append(tend - tstart)
                else:
                    self.bico.time[self.level] += tend - tstart
        if len(candidates) == 0:
            if self.bico.verbose:
                logger.debug("No nearest neighbor found.")
            self.num_cfs += 1
            self.nn_engine.insert_candidate(point=point_cf.ref.p, metadata=self.num_cfs)
            # self.ann_engine.store_vector(point_cf.ref.p, data=self.num_cfs)
            new_node = BICONode(self.level + 1, self.dim, self.proj, self.bico, self.projection_func)
            # new_node.cf = ClusteringFeature(geometry, geometry, geometry*geometry, 1)
            new_node.cf = point_cf
            # debug
            if len(self.point_to_biconode) != self.num_cfs - 1:
                logger.error("Something is wrong: {} != {}".format(len(self.point_to_biconode), self.num_cfs - 1))
            self.point_to_biconode.append(new_node)
            return 1
        else:
            if self.bico.verbose:
                logger.debug(str(len(candidates)) + " nearest neighbor found!")
                logger.debug(candidates)
            nearest = candidates[0]
            node = nearest.data  # contains the index
            # sanity check
            if len(self.point_to_biconode) < node - 2:
                logger.error("Something is wrong: {} > {}".format(len(self.point_to_biconode), node - 2))
            return self.point_to_biconode[node - 1].insert_point(point_cf)

    def output_cf(self, f: TextIO) -> None:
        if self.level > 0:
            f.write(str(self.cf) + "\n")
        for node in self.point_to_biconode:
            node.output_cf(f)

    def get_cf(self) -> List[np.ndarray]:
        cur = []
        if self.level > 0:
            cur.append(np.insert(self.cf.center().p, 0, self.cf.size))
        for node in self.point_to_biconode:
            cur = cur + node.get_cf()
        return cur
