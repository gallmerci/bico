from collections import deque
from math import sqrt

import logging
import numpy as np
from bico.geometry.point import Point
from bico.nearest_neighbor.base import NearestNeighbor
from bico.nearest_neighbor.random_binary_projections import RandomBinaryNN
from bico.nearest_neighbor.random_binary_tree import RandomBinaryTreeNN
from bico.nearest_neighbor.simple_projection import SimpleProjection
from bico.utils.BICONode import BICONode
from bico.utils.ClusteringFeature import ClusteringFeature
from datetime import datetime

logger = logging.getLogger(__name__)


class BICO:
    """ Base class for bico applications """

    def __init__(self, dimension: int, number_projections: int, coreset_size: int,
                 projection_method='simple', verbose=False, track_time=False):
        """
        :param dimension:
            Dimension of input points
        :param number_projections:
            Number of projections for faster nearest neighbor search. Time trade-off: The more projections the less distances to
            potential neighbors are computed. But more projections also mean more space and insertion time.
        :param coreset_size:
            Maximum number of points of the reduction result
        :param projection_method:
            Projection method to accelerate nearest neighbor search. Possible values are:
            - 'simple': Simple random projection technique (default)
            - 'binary': Random binary projection technique implement by nearpy package
            - 'binary_tree': Random binary tree technique implement by nearpy package
        :param track_time:
            activate tracking of nearest neighbor time consumption on each level of the BICO tree
        :param verbose:
            activate debug logging
        """
        self.dimension = dimension
        self.number_projections = number_projections
        self.projection_func = projection_method
        self.coreset_size = coreset_size
        self.thresh = 1
        self.num_cfs = 0
        self.buffer_phase = True
        self.buffer = []
        self.time = []
        self.track_time = track_time
        self.verbose = verbose

        name = 'create_{}_projection'.format(projection_method.lower())
        self.projection_func = getattr(BICO, name, None)
        if self.projection_func is None:
            raise ValueError('Unknown projection method: {}'.format(projection_method))

        self.root = BICONode(0, dimension, number_projections, self, projection_func=self.projection_func)

    @staticmethod
    def create_simple_projection(dim: int, proj: int, thresh: float) -> NearestNeighbor:
        return SimpleProjection(dim, proj, thresh)

    @staticmethod
    def create_binary_projection(dim: int, proj: int, thresh: float) -> NearestNeighbor:
        return RandomBinaryNN(dim, proj, thresh)

    @staticmethod
    def create_binary_tree_projection(dim: int, proj: int, thresh: float) -> NearestNeighbor:
        return RandomBinaryTreeNN(dim, proj, thresh)

    def insert_point(self, point: Point):
        """
        Insert a single geometry into the data structure.
        :param point:
            Point to be inserted
        :return:
            None
        """
        if self.verbose:
            logger.debug("Insert point: {}".format(point))
        if self.buffer_phase:
            self.buffer.append(point)
            if len(self.buffer) > sqrt(self.coreset_size):
                logger.info("Buffer phase finished.")
                self.buffer_phase = False
                minDist = -1
                for p in self.buffer:
                    for p2 in self.buffer:
                        d = p.p - p2.p
                        dist = np.ma.inner(d, d)
                        if dist == 0:
                            continue
                        if minDist == -1 or minDist > dist:
                            minDist = dist
                if self.verbose:
                    logger.debug("Initial Threshold: " + str(16 * minDist))
                self.thresh = 16 * minDist
                for p in self.buffer:
                    self.insert_point(p)
        else:
            point_cf = ClusteringFeature(point, point, point * point, 1)
            self.num_cfs += self.root.insert_point(point_cf)
            if self.num_cfs > self.coreset_size:
                # rebuild
                tstart = datetime.now()
                queue = deque()
                for node in self.root.point_to_biconode:
                    queue.append(node)
                self.num_cfs = 0
                self.thresh *= 2.0
                self.rebuild_tree(queue)
                tend = datetime.now()
                logger.info("Rebuild time: {}".format(tend - tstart))

    def rebuild_tree(self, queue: deque):
        """
        Rebuilds the BICO tree based on the clustering features in the input queue
        :param queue:
            Queue contains the roots of other BICO trees where all clustering features
            in these trees are inserted into this new tree
        :return:
            None
        """
        self.root = BICONode(0, self.dimension, self.number_projections, self, self.projection_func)
        logger.info(
            "Created too many coreset points. Start rebuilding with new threshold: {}".format(self.thresh))
        while len(queue) > 0:
            node1 = queue.pop()
            for node2 in node1.point_to_biconode:
                queue.append(node2)
            self.num_cfs += self.root.insert_point(node1.cf)

    def get_threshold(self, level: int) -> float:
        """
        Returns internal threshold for a specified level of the internal tree data structure.
        """
        return self.thresh

    def get_radius(self, level: int) -> float:
        """
        Returns internal radius for the neighborhood of a geometry in a specified level of the internal tree data structure.
        """
        # print "getR: " + str(self.thresh / float(1 << (3+level)))
        return self.thresh / float(1 << (3 + level))

    def output_coreset(self, file_name: str):
        """
        Recursive output of the reduced geometry set.
        :param file_name:
            File name
        :return:
            None
        """
        coreset = self.get_coreset()
        np.save(file_name, coreset)

    def get_coreset(self) -> np.ndarray:
        """
        Returns reduced data set
        :return:
            Returns (coreset size) x (dim+1) dimensional numpy array where each row contains the weight / size in the first column
            and the reference geometry of a clustering feature in the remaining columns.
        """
        return np.vstack(self.root.get_cf())
