from abc import ABC, abstractmethod

import numpy as np


class TrajectoryBase(ABC):
    """
    Base trajectory interface acting as a blueprint for classes that inherit from it.
    """
    def __init__(self, trajectory: np.ndarray):
        self._trajectory = trajectory

    @property
    @abstractmethod
    def joints(self) -> np.ndarray:
        raise NotImplementedError
