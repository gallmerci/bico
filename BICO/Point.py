import numpy as np


class Point:
    """
    Point class for point representation in BICO data structures
    """

    def __init__(self, point):
        """
        :param point:
            1-D Numpy array representing a point
        """
        self.p = point

    def set_point(self, point):
        self.p = point

    def __add__(self, other):
        return Point(self.p + other.p)

    def __iadd__(self, other):
        self.p += other.p
        return self

    def __mul__(self, other):
        """
        Inner product of two points
        :param other:
            Second point for inner product computation
        :return:
            Result of inner product as a float.
        """
        return np.ma.inner(self.p, other.p)

    def scalar_mul(self, scalar):
        """
        Scalar multiplication of this point with a scalar.
        :param scalar:
            Scalar as a float
        :return:
            New point with the result of the scalar multiplication
        """
        return Point(scalar * self.p)

    def __str__(self):
        return str(self.p)
