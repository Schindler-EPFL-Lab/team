import json
import os

import numpy as np

from team.timed_trajectory import TimedTrajectory


class Trajectory(TimedTrajectory):
    """Holds the data recorded by the robot during a trajectory.

    :param: trajectory: a numpy array with timestamps, joint position, and tcp position.
    """

    def __init__(self, trajectory: np.ndarray):
        if np.shape(trajectory)[1] != 10:
            raise RuntimeError("array dimensions are not correct")
        super().__init__(trajectory, False)

    @classmethod
    def from_file(cls, trajectory_path: str) -> "Trajectory":
        """Creates a new Trajectory object from the data in a json file.

        :param trajectory_path: path to the json file
        :return: the trajectory containing the json file information
        """
        # check that the file format is .json
        assert os.path.splitext(trajectory_path)[1] == ".json", (
            "Erroneous file format." ".json format required " "for reading the file"
        )
        with open(trajectory_path, "r") as f:
            trajectory_dict = json.load(f)
            d = np.transpose(np.array(list(trajectory_dict.values())))
            timestamp = d[:, 0].reshape(-1, 1)
            joints = d[:, -6:]
            tcp_position = d[:, 1:4]
            joints_info = np.hstack((timestamp, joints, tcp_position))
            return cls(joints_info)

    @property
    def tcp(self) -> np.ndarray:
        return self._trajectory[:, 7:10]

    @property
    def joint_and_tcp(self) -> np.ndarray:
        return self._trajectory[:, 1:]
