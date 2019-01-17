from abc import abstractmethod, ABC
import numpy as np
import attr
from typing import Any, List


@attr.s
class NearestNeighborResult:
    point = attr.ib(validator=attr.validators.instance_of(np.ndarray))
    data = attr.ib()
    distance = attr.ib(validator=attr.validators.instance_of(float))


class NearestNeighbor(ABC):
    """ Abstract class for nearest neighbor implementation """
    @abstractmethod
    def get_candidates(self, point: np.ndarray) -> List[NearestNeighborResult]:
        """
        Get nearest neighbor candidates for a single point
        :param point:
            Point represented by 1-D numpy array
        :return:
            List of nearest neighbor candidates
        """
        pass

    @abstractmethod
    def insert_candidate(self, point: np.ndarray, metadata: Any):
        """
        Insert a single geometry into the data structure.
        :param point:
            Point represented by 1-D numpy array
        :param metadata:
            Arbitrary metadata that can be attached to that point
        :return:
            None
        """
        pass
