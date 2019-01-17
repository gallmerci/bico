import numpy as np


class Point:
    """
    Point class for geometry representation in bico data structures
    """

    def __init__(self, point: np.ndarray):
        """
        :param point:
            1-D Numpy array representing a geometry
        """
        self.p = point

    def set_point(self, point: np.ndarray):
        self.p = point

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.p + other.p)

    def __iadd__(self, other: 'Point') -> 'Point':
        self.p += other.p
        return self

    def __mul__(self, other: 'Point') -> np.ndarray:
        """
        Inner product of two points
        :param other:
            Second geometry for inner product computation
        :return:
            Result of inner product as a float.
        """
        return np.inner(self.p, other.p)

    def scalar_mul(self, scalar: float) -> 'Point':
        """
        Scalar multiplication of this geometry with a scalar.
        :param scalar:
            Scalar as a float
        :return:
            New geometry with the result of the scalar multiplication
        """
        return Point(scalar * self.p)

    def __str__(self) -> str:
        return str(self.p)
