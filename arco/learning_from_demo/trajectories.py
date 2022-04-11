import numpy as np

from arco.learning_from_demo.data_preprocessing import DataPreprocessing


class Trajectories:
    """
    Loads data from a dataset of recorded demonstrations and preprocesses them with
    the from_dataset_file method or directly loads the trajectories data to analyse

    :param aligned_trajectories: preprocessed trajectories
           data has shape (nb_trajectories x demo_length x nb_joints + 1)
           the first column of each sample denotes the timestamp
    """
    def __init__(self, aligned_trajectories: np.ndarray):
        self.aligned_trajectories = aligned_trajectories
        self.timestamps = self.aligned_trajectories[0, :, 0]
        self.joints_trajectories = self.aligned_trajectories[:, :, 1:]

    @classmethod
    def from_dataset_file(cls, data_path: str) -> "Trajectories":
        """
        Loads data from dataset, preprocesses it and returns the aligned trajectories

        :param data_path: path to the dataset
        :return: aligned trajectories
        """
        dp = DataPreprocessing(data_path, sampling_rate=100)
        dp.preprocessing()
        aligned_trajectories = dp.aligned_and_padded_trajectories
        return cls(np.array(aligned_trajectories))
