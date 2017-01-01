from nearpy import Engine
from nearpy.hashes import RandomBinaryProjectionTree
from nearpy.hashes import RandomBinaryProjections
from squared_euclidean import SquaredEuclideanDistance
from nearpy.filters import DistanceThresholdFilter
from ClusteringFeature import ClusteringFeature
from Point import Point
import numpy as np
from datetime import datetime


class BICONode:
    def __init__(self, level, dim, proj, bico):
        self.level = level
        self.dim = dim
        self.proj = proj
        self.point_to_biconode = {}
        self.rbpt = RandomBinaryProjectionTree('rbpt', proj, 1)
        self.rbp = RandomBinaryProjections('rbp', proj)
        self.sqdist = SquaredEuclideanDistance()
        self.ann_engine = Engine(dim, lshashes=[self.rbp], distance=self.sqdist,
                                 vector_filters=[DistanceThresholdFilter(bico._getR(self.level))])
        self.num_cfs = 0
        self.bico = bico
        self.cf = ClusteringFeature(Point(np.zeros(dim)), Point(np.zeros(dim)), 0, 0)

    def insert_point(self, point_cf):
        # print "Insert point " + str(point)
        # check whether point fits into CF
        if self.level > 0:
            if self.cf.size == 0:
                # print "Put point into CF"
                # self.cf += ClusteringFeature(point, point, point*point, 1)
                # self.cf.ref = point
                self.cf += point_cf
                self.cf.ref = point_cf.ref
            else:
                # print str(self.cf)
                # testCF = ClusteringFeature(point, point, point*point, 1)
                # print testCF
                test = self.cf + point_cf
                # print test
                cost = test.kmeans_cost(self.cf.ref)
                # print "Cost: " + str(cost) + ", Thresh: " + str(self.bico.getT(self.level))
                if cost < self.bico._getT(self.level):
                    # print "Put point into CF"
                    self.cf = test
                    return 0
                    # print "Put does not fit into CF"
                    # print "Point does not fit into CF"

        # search nearest neighbor and insert point there or open new BICONode
        candidates = []
        if self.num_cfs > 0:
            tstart = datetime.now()
            candidates = self.ann_engine.neighbours(point_cf.ref.p)
            tend = datetime.now()
            if len(self.bico.time) < self.level + 1:
                self.bico.time.append(tend - tstart)
            else:
                self.bico.time[self.level] += tend - tstart
        if len(candidates) == 0:
            # print "Keine nearest neighbor gefunden"
            self.num_cfs += 1
            self.ann_engine.store_vector(point_cf.ref.p, data=self.num_cfs)
            new_node = BICONode(self.level + 1, self.dim, self.proj, self.bico)
            # new_node.cf = ClusteringFeature(point, point, point*point, 1)
            new_node.cf = point_cf
            # print "ref: " + str(point)
            self.point_to_biconode[self.num_cfs] = new_node
            return 1
        else:
            # print str(len(candidates)) + " nearest neighbor gefunden!"
            # print candidates
            nearest = candidates[0]
            node = nearest[1]
            return self.point_to_biconode[node].insert_point(point_cf)

    def output_cf(self, f):
        if self.level > 0:
            f.write(str(self.cf) + "\n")
        for (key, node) in self.point_to_biconode.iteritems():
            node.output_cf(f)
