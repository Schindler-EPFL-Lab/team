from typing import Optional
import os

import json
import numpy as np
import pandas as pd


def create_default_dict(keys: Optional[list[str]] = None):
    """
    Function to create default dictionary to store trajectory demonstration.
    :param keys: optional dictionary keys
    :return: dictionary with values initialized to empty lists
    """
    if not keys:
        keys = [
            "timestamp",
            "tcp_x",
            "tcp_y",
            "tcp_z",
            "tcp_q1",
            "tcp_q2",
            "tcp_q3",
            "tcp_q4",
            "cf1",
            "cf4",
            "cf6",
            "cfx",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]
    return {key: [] for key in keys}


def read_json_file(filename_path: str) -> pd.DataFrame:
    """
    Read json file in a pandas dataframe
    :param filename_path: filename path
    :return: the dataframe
    """
    df = pd.read_json(filename_path)
    return df


def target_encoding(dataframe: pd.DataFrame, index: int) -> str:
    """
    Receives the dataframe and the row index to extract the target data.
    Finally, it constructs the message and it encodes it as a string.
    :param dataframe: dataframe to read data from
    :param index: row index
    :return: the target pose in the right encoding
    """
    target = dataframe.iloc[index, 1:12].to_list()
    pos_list = target[0:3]
    ori_list = target[3:7]
    config_list = target[7:11]
    ext_axis_list = [9e9, 9e9, 9e9, 9e9, 9e9, 9e9]
    return str([pos_list, ori_list, config_list, ext_axis_list])


def get_demo_files() -> list[str]:
    """
    This function retrieves all the demonstrations files in the demonstrations folder.
    All the files that are not in the .json format are discarded.
    :return: a list containing all file paths corresponding to demonstrations
    """
    repo_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    demo_path = os.path.join(repo_dir, "demonstrations")
    # initialize empty list to store demonstrations files paths
    list_file_paths = []
    # explore all the files present/in all folders inside the parent folder
    for (dir_path, dir_names, file_names) in os.walk(demo_path):
        for file in file_names:
            # discard if not a .json file
            if file.endswith(".json"):
                list_file_paths.append(os.path.join(dir_path, file))
    return list_file_paths


def check_data_timestamps(list_file_paths: list[str]) -> None:
    """
    The function controls that the timestamps are updated at each single data reading,
    if it is not the case, an assertion error is thrown
    :param list_file_paths:
    :return: None
    """
    # analyse each single file
    for data_path in list_file_paths:
        with open(data_path, 'r') as f:
            # load data and extract time vector
            data = json.load(f)
            time = np.array(data["timestamp"])
            delta_t = time[1:] - time[0:-1]
            # check that time is updated at every data reading step
            assert sum(delta_t == 0) == 0, "Timestamps not updated correctly"


def check_nan_values(list_file_paths: list[str]) -> None:
    """
    The function checks that each data dictionary in the dataset doesn't contain missing
    values, if it is not the case, an assertion error is thrown
    :param list_file_paths:
    :return: None
    """
    # consider empty string and numpy.inf as na values
    pd.set_option('mode.use_inf_as_na', True)
    # analyse each single file
    for data_path in list_file_paths:
        # load data as pandas dataframe
        df = pd.read_json(data_path)
        assert not df.isnull().values.any(), "Missing values in the dictionary"


def check_reading_files(list_file_paths: list[str]) -> None:
    """
    The function verifies that each data dictionary in the dataset has the required keys
    ,if it is not the case, an assertion error is thrown
    :param list_file_paths:
    :return: None
    """
    # analyse each single file
    for data_path in list_file_paths:
        # load data as pandas dataframe
        df = pd.read_json(data_path)
        default_dict = create_default_dict()
        # checks that the dataframe has the same keys as the recorded dictionary
        for df_key, dict_key in zip(df.keys(), default_dict.keys()):
            assert df_key == dict_key, "Dictionary keys inconsistent"
