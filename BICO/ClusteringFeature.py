class ClusteringFeature:
    """
    Main component of BICO data structure. A clustering feature consists of a reference point, the sum (and squared sum) of points
    inserted into this feature, and the number of inserted points.
    """

    def __init__(self, ref, sum, squared, size):
        self.sum = sum
        self.squared = squared
        self.size = size
        self.ref = ref

    def __add__(self, other):
        """
        Insert new point to the clustering feature
        :param other:
            Point to be inserted
        :return:
            ClusteringFeature with point inserted
        """
        return ClusteringFeature(self.ref, self.sum + other.sum, self.squared + other.squared, self.size + other.size)

    def __iadd__(self, other):
        self.sum += other.sum
        self.squared += other.squared
        self.size += other.size
        return self

    def center(self):
        """
        Returns the centroid of the inserted points
        :return:
            Point representing the centroid
        """
        return self.sum.scalar_mul(float(1) / self.size)

    def kmeans_cost(self, center):
        """
        Returns the 1-means cost of the inserted points to the specified center
        :param center:
            Center of the 1-means solution
        :return:
            Cost of the solution for the inserted points
        """
        return self.squared - 2 * (center * self.sum) + self.size * (center * center)

    def __str__(self):
        return str(self.size) + " " + " ".join(map(str, self.center().p.tolist()))
        # return str(self.size) + " " + str(self.center().p)
        # return "Summary CF\nSum: " + str(self.sum)+ "\nSS: " + str(self.squared) + "\nSize: " + str(self.size) + "\nRef: " + str(self.ref)
