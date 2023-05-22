import json
import os

import numpy as np
from scipy import interpolate

from team.trajectory_base import TrajectoryBase


class TimedTrajectory(TrajectoryBase):
    """Holds the data recorded by the robot during a trajectory.

    :param: trajectory: a numpy array with timestamps and joint position.
    """

    def __init__(self, trajectory: np.ndarray, check_dim: bool = True):
        if check_dim and np.shape(trajectory)[1] != 7:
            raise RuntimeError("array dimensions are not correct")
        super().__init__(trajectory)
        self.sampling_rate = 1 / (self._trajectory[1, 0] - self._trajectory[0, 0])
        self.dt = self._trajectory[1, 0] - self._trajectory[0, 0]

    @classmethod
    def from_file(cls, trajectory_path: str) -> "TimedTrajectory":
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
            joints_info = np.hstack((timestamp, joints))
            return cls(joints_info)

    @property
    def timestamps(self) -> np.ndarray:
        return self._trajectory[:, 0]

    @property
    def joints(self) -> np.ndarray:
        return self._trajectory[:, 1:7]

    @property
    def joint_timestamps(self) -> np.ndarray:
        return self._trajectory[:, :7]

    @property
    def period(self) -> float:
        """Average time elapse between two measurements."""
        average_time = (self.timestamps[-1] - self.timestamps[0]) / (
            len(self.timestamps) - 1
        )
        return float(average_time)

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
            self.timestamps, self._trajectory[:, 1:], axis=0
        )
        num = (
            round((self.timestamps[-1] - 0) / (1 / des_freq)) + 1
        )  # i.e. length of resulting array
        time_new = np.linspace(0, self.timestamps[-1], num)
        # use interpolation function returned by `interp1d` to generate new data points
        joints_new = f_interpolate(time_new)
        self._trajectory = np.append(time_new.reshape(-1, 1), joints_new, axis=1)

    def pad_end_to(self, final_len: int) -> None:
        """Pads the end of the trajectory by duplicating the last element until [self]
        is of len [final_len].

        Increment the timestamps by 0.01.

        :param final_len: the final length of the padded trajectory
        """
        if len(self) != final_len:
            nb_samples = final_len - len(self) + 1
            current_len = len(self)

            super().pad_end_to(final_len)

            self.trajectory[current_len - 1 : len(self), 0] += np.arange(
                0, nb_samples * 0.01, 0.01
            )
