import json
import os

import numpy as np
import pandas as pd
from scipy import interpolate

from arco.learning_from_demo.trajectory_base import TrajectoryBase


class Trajectory(TrajectoryBase):
    """Holds the data recorded by the robot during a trajectory.

    :param: trajectory: a numpy array with timestamps, joint position, and tcp position.
    """

    def __init__(self, trajectory: np.ndarray):
        assert np.shape(trajectory)[1] == 10, "array dimensions are not correct"
        super().__init__(trajectory)

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
    def trajectory(self) -> np.ndarray:
        return self._trajectory

    @property
    def timestamps(self) -> np.ndarray:
        return self._trajectory[:, 0]

    @property
    def joints(self) -> np.ndarray:
        return self._trajectory[:, 1:7]

    @property
    def tcp(self) -> np.ndarray:
        return self._trajectory[:, 7:10]

    @property
    def joint_and_tcp(self) -> np.ndarray:
        return self._trajectory[:, 1:]

    @property
    def joint_timestamps(self) -> np.ndarray:
        return self._trajectory[:, :7]

    @property
    def average_sampling(self) -> float:
        return len(np.unique(self._trajectory, axis=0)) / self.timestamps[-1]

    def upsample(self, des_freq: int) -> None:
        """
        Upsample the trajectory by computing a linear interpolation function fitting
        the joints to the timestamp and then generates the missing values applying
        the interpolating function to the time vector sampled at des_freq.

        :param des_freq: the desired sampling frequency
        """

        # Starts the timestamps at 0
        self._trajectory[:, 0] = self.timestamps - self.timestamps[0]

        f_interpolate = interpolate.interp1d(
            self.timestamps, self.joint_and_tcp, axis=0
        )
        num = (
            round((self.timestamps[-1] - 0) / (1 / des_freq)) + 1
        )  # i.e. length of resulting array
        time_new = np.linspace(0, self.timestamps[-1], num)
        # use interpolation function returned by `interp1d` to generate new data points
        joints_new = f_interpolate(time_new)
        self._trajectory = np.append(time_new.reshape(-1, 1), joints_new, axis=1)

    def pad_end(self, final_len: int) -> None:
        """Pads the end of the trajectory by duplicating the last element until [self]
        is of len [final_len].

        :param final_len: the final length of the padded trajectory
        """
        if len(self) != final_len:
            # computes number of missing data points
            nb_samples = final_len - len(self)
            # duplicates nb_samples time the last row
            df = pd.DataFrame(self._trajectory)
            df = pd.concat([df, pd.concat([df.iloc[-1:]] * nb_samples)])
            # updates the timestamps
            df.iloc[-nb_samples:, 0] = df.iloc[-nb_samples:, 0] + np.linspace(
                0.01, 0.01 * nb_samples, nb_samples
            )
            df.reset_index(drop=True, inplace=True)
            self._trajectory = df.to_numpy()
