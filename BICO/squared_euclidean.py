import numpy
import scipy

from nearpy.distances.distance import Distance
from scipy.spatial.distance import pdist
from datetime import datetime
import scipy.weave as weave


class SquaredEuclideanDistance(Distance):
    """ Squared Euclidean distance """

    def distance(self, x, y):
        """
        Computes squared euclidean distance between vectors x and y. Returns float.
        """

        d = x - y
        # dist = numpy.ma.inner(d,d)
        dist = numpy.sum(d ** 2)
        # dist = pdist([x,y], 'sqeuclidean')
        # n = len(x)
        # code = \
        #     """
        #     int i;
        #     double sum = 0.0, delta = 0.0f;
        #     for (i = 0; i < n; i++) {
        #         delta = (x[i]-y[i]);
        #         sum += delta*delta;
        #     }
        #     return_val = sum;
        #     """
        # dist = weave.inline(code, ['x', 'y', 'n'])
        return dist
