from __future__ import annotations
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

    def get_joint(self, i: int) -> np.ndarray:
        return self.joints[:, i]

    def get_joints_at_index(self, i: int) -> np.ndarray:
        """Return all joint position at index [i]

        :param i: index of the joints in the [self._trajectory]
        :return: the joints in an array of size (1, nb of joints)
        """
        return self.joints[i, :]

    def __len__(self) -> int:
        return len(self._trajectory)

    def rms_cumulative_error(self, other_trajectory: TrajectoryBase) -> float:
        """
        Compute the root squared cumulative error along the motion between the
        trajectory to track and the executed trajectory

        :param other_trajectory: trajectory to track, input to the joint position
                                 controller
        :return: the cumulative error along the trajectory
        """

        error = 0
        for i in range(len(self.joints)):
            error += np.linalg.norm(
                self.get_joints_at_index(i) - other_trajectory.get_joints_at_index(i)
            )
        return error
