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

    def joints_to_string(self, tol_diff: int = 1) -> str:
        """
        Down-sample the trajectory based on the distance between points and encode the
        down-sample trajectory into a string.
        The points are selected based on a distance threshold.

        :param tol_diff: threshold on the error norm between consecutive targets
        :return: string containing the sequence of joint targets separated by a new line
        """

        current_pose = self.get_joints_at_index(0)
        selected_waypoint = [
            self._joint_to_string(self.get_joints_at_index(0))
        ]
        for joints in self.joints:
            if self._target_is_eligible(joints, current_pose, tol_diff):
                selected_waypoint.append(self._joint_to_string(joints))
                current_pose = joints
        return "\n".join(selected_waypoint)

    @staticmethod
    def _joint_to_string(joints: np.ndarray) -> str:
        """
        Converts the joints vector into a string

        :param joints: joint angles vector to cast
        :return: string containing the joints vector
        """
        return str(np.around(joints, decimals=3).tolist())

    @staticmethod
    def _target_is_eligible(
        next_target: np.ndarray, current_pose: np.ndarray, tol_diff: float
    ) -> bool:
        """
        Checks that the distance between the actual robot joint angles and the target
        joint angles is large enough. It avoids performing micro movements.

        :param next_target: vector of the next target joint angles
        :param current_pose: vector of current joint angles
        :param tol_diff: tolerance on the error norm between consecutive targets
        :return: boolean assessing if the target needs to be considered or not
        """
        distance = np.linalg.norm(next_target - current_pose)
        return distance > tol_diff
