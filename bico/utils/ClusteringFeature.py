from bico.geometry.point import Point


class ClusteringFeature:
    """
    Main component of bico data structure. A clustering feature consists of a reference geometry, the sum (and squared sum) of points
    inserted into this feature, and the number of inserted points.
    """

    def __init__(self, ref: Point, sum: Point, squared: float, size: int):
        """
        :param ref:
            Reference point stored for this clustering feature
        :param sum:
            Sum of all points in the clustering feature
        :param squared:
            Sum of self inner products for all points
        :param size:
            Number of points represented by this clustering feature
        """
        self.sum = sum
        self.squared = squared
        self.size = size
        self.ref = ref

    def __add__(self, other: 'ClusteringFeature') -> 'ClusteringFeature':
        """
        Insert new geometry to the clustering feature
        :param other:
            Point to be inserted
        :return:
            ClusteringFeature with geometry inserted
        """
        return ClusteringFeature(self.ref, self.sum + other.sum, self.squared + other.squared, self.size + other.size)

    def __iadd__(self, other: 'ClusteringFeature') -> 'ClusteringFeature':
        self.sum += other.sum
        self.squared += other.squared
        self.size += other.size
        return self

    def center(self) -> Point:
        """
        Returns the centroid of the inserted points
        :return:
            Point representing the centroid
        """
        return self.sum.scalar_mul(float(1) / self.size)

    def kmeans_cost(self, center) -> float:
        """
        Returns the 1-means cost of the inserted points to the specified center
        :param center:
            Center of the 1-means solution
        :return:
            Cost of the solution for the inserted points
        """
        return self.squared - 2 * (center * self.sum) + self.size * (center * center)

    def __str__(self) -> str:
        return str(self.size) + " " + " ".join(map(str, self.center().p.tolist()))
