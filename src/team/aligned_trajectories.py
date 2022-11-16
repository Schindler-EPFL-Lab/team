from typing import Optional

import numpy as np

from team.data_preprocessing import DataPreprocessing
from team.trajectory import Trajectory
from team.utility.handling_data import get_demo_files


class AlignedTrajectories:
    """
    loads data from a dataset of recorded demonstrations and preprocesses them
              with the load_dataset_and_preprocess class method. The trajectories are
              given by the preprocessed data

    :param aligned_trajectories: preprocessed trajectories
           data has shape (nb_trajectories x demo_length x nb_joints + 1)
           the first column of each trajectory denotes the timestamp
    """

    def __init__(self, aligned_trajectories: np.ndarray):
        self.aligned_trajectories = aligned_trajectories

    @classmethod
    def from_list_trajectories(
        cls, trajectories: list[Trajectory]
    ) -> "AlignedTrajectories":
        """Creates an AlignedTrajectories from a list of Trajectories.

        :param trajectories: a list of trajectories
        :return: an AlignedTrajectories containing all the trajectories of
            [trajectories]
        """
        np_trajectories = np.stack(([traj.joint_timestamps for traj in trajectories]))
        return cls(np_trajectories)

    @property
    def timestamps(self):
        return self.aligned_trajectories[0, :, 0]

    @property
    def joints_trajectories(self):
        return self.aligned_trajectories[:, :, 1:]

    def get_trajectory(self, i: int) -> Trajectory:
        return Trajectory(self.aligned_trajectories[i, :, :])

    @staticmethod
    def _load_data(data_path: str) -> list[Trajectory]:
        """
        Loads the dataset as a list of dataframes

        :param data_path: the path to reach the dataset folder
        :return: the list of dataframes corresponding to the demonstration dataset
        """
        files_paths = get_demo_files(data_path)
        trajectories = []
        for file_path in files_paths:
            trajectories.append(Trajectory.from_file(file_path))
        return trajectories

    @classmethod
    def load_dataset_and_preprocess(
        cls, data_path: str, window: Optional[float] = None
    ) -> "AlignedTrajectories":
        """
        Loads data from dataset, preprocesses it and returns the aligned trajectories

        :param data_path: path to the dataset
        :param window: only allow for maximal shifts from the two diagonals smaller
        than this number in seconds. Default to maximum.
        :return: aligned trajectories
        """
        trajectories_list = AlignedTrajectories._load_data(data_path)
        dp = DataPreprocessing(trajectories_list, sampling_rate=100)
        dp.preprocessing(window)
        return cls.from_list_trajectories(dp.aligned_and_padded_trajectories)
