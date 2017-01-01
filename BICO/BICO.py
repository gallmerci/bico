from BICONode import BICONode
from collections import deque
from ClusteringFeature import ClusteringFeature
import numpy
from math import sqrt


class BICO:
    """ Base class for BICO applications """

    def __init__(self, dim, n, p, coreset):
        """
        :param dim:
            Dimension of input points
        :param n:
            Total number of input points
        :param p:
            Number of projections for faster nearest neighbor search. Time trade-off: The more projections the less distances to
            potential neighbors are computed. But more projections also mean more space and insertion time.
        :param coreset:
            Maximum number of points of the reduction result
        """
        self.dim = dim
        self.n = n
        self.p = p
        # self.k = k
        self.coreset = coreset
        self.thresh = 1
        self.num_cfs = 0
        self.root = BICONode(0, dim, p, self)
        self.buffer_phase = True
        self.buffer = []
        self.time = []

    def insert_point(self, point):
        """
        Insert a single point into the data structure.
        :param point:
            Point represented by 1-D numpy array
        :return:
            None
        """
        if self.buffer_phase:
            self.buffer.append(point)
            if len(self.buffer) > sqrt(self.coreset):
                self.buffer_phase = False
                minDist = -1
                for p in self.buffer:
                    for p2 in self.buffer:
                        d = p.p - p2.p
                        dist = numpy.ma.inner(d, d)
                        if dist == 0:
                            continue
                        if minDist == -1 or minDist > dist:
                            minDist = dist
                print "Initial Threshold: " + str(16 * minDist)
                self.thresh = 16 * minDist
                for p in self.buffer:
                    self.insert_point(p)
        else:
            point_cf = ClusteringFeature(point, point, point * point, 1)
            self.num_cfs += self.root.insert_point(point_cf)
            if self.num_cfs > self.coreset:
                # rebuild
                queue = deque()
                for (key, node) in self.root.point_to_biconode.iteritems():
                    queue.append(node)
                self.num_cfs = 0
                self.thresh *= 2.0
                self.root = BICONode(0, self.dim, self.p, self)
                print "Rebuilding... Threshold: " + str(self.thresh)
                while len(queue) > 0:
                    node1 = queue.pop()
                    for (key, node2) in node1.point_to_biconode.iteritems():
                        queue.append(node2)
                    self.num_cfs += self.root.insert_point(node1.cf)

    def _getT(self, level):
        """
        Returns internal threshold for a specified level of the internal tree data structure.
        """
        return self.thresh

    def _getR(self, level):
        """
        Returns internal radius for the neighborhood of a point in a specified level of the internal tree data structure.
        """
        # print "getR: " + str(self.thresh / float(1 << (3+level)))
        return self.thresh / float(1 << (3 + level))

    def output_coreset(self, f):
        """
        Recursive output of the reduced point set.
        :param f:
            File descriptor
        :return:
            None
        """
        self.root.output_cf(f)
