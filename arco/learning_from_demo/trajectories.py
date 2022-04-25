import json
import os

import numpy as np
import pandas as pd

from arco.learning_from_demo.data_preprocessing import DataPreprocessing
from arco.utility.handling_data import get_demo_files


class Trajectories:
    """
    Option 1: loads data from a dataset of recorded demonstrations and preprocesses them
              with the load_dataset_and_preprocess class method. The trajectories are
              given by the preprocessed data
    Option 2: loads a single trajectory data to analyse from a .json file with the
              load_single_trajectory class method. Tha trajectory is directly given by
              the data, no preprocessing since only a single trajectory is considered

    :param aligned_trajectories: preprocessed trajectories
           data has shape (nb_trajectories x demo_length x nb_joints + 1)
           the first column of each trajectory denotes the timestamp
    """
    def __init__(self, aligned_trajectories: np.ndarray):
        self.aligned_trajectories = aligned_trajectories
        self.timestamps = self.aligned_trajectories[0, :, 0]
        self.joints_trajectories = self.aligned_trajectories[:, :, 1:]

    @staticmethod
    def _load_data(data_path: str) -> list[pd.DataFrame]:
        """
        Loads the dataset as a list of dataframes

        :param data_path: the path to reach the dataset folder
        :return: the list of dataframes corresponding to the demonstration dataset
        """
        files_paths = get_demo_files(data_path)
        df_list = []
        for file_path in files_paths:
            df_list.append(pd.read_json(file_path))
        return df_list

    @classmethod
    def load_dataset_and_preprocess(cls, data_path: str) -> "Trajectories":
        """
        Loads data from dataset, preprocesses it and returns the aligned trajectories

        :param data_path: path to the dataset
        :return: aligned trajectories
        """
        trajectories_list = Trajectories._load_data(data_path)
        dp = DataPreprocessing(trajectories_list, sampling_rate=100)
        dp.preprocessing()
        aligned_trajectories = dp.aligned_and_padded_trajectories
        return cls(np.array(aligned_trajectories))

    @classmethod
    def load_single_trajectory(cls, trajectory_path: str) -> "Trajectories":
        """
        Loads single trajectory with the aim to playback the demonstration

        :param trajectory_path: path to the trajectory file
        :return: trajectory
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
            len_demo, nb_features = np.shape(joints_info)
            return cls(joints_info.reshape((1, len_demo, nb_features)))
