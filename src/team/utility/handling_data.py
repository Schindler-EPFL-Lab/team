from pathlib import Path

import numpy as np
import pandas as pd


def create_default_dict() -> dict[str, list]:
    """
    Creates default dictionary to store trajectory demonstration.

    :return: dictionary with values initialized to empty lists
    """
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
    Reads json file in a pandas dataframe.

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
    :return: the target joints angles in the right encoding
    """
    joints = dataframe.iloc[index, 12:].to_list()
    ext_axis_list = [9e9, 9e9, 9e9, 9e9, 9e9, 9e9]
    return str([joints, ext_axis_list])


def check_data_timestamps(data: dict) -> None:
    """
    Controls that the timestamps are updated at each single data reading.
    If consecutive timestamps are equal, an error is raised.

    :param data:the data dictionary to analyse
    :raises ValueError: consecutive timestamp readings are not updated correctly
    """
    time = np.array(data["timestamp"])
    delta_t = time[1:] - time[0:-1]
    # check that time is updated at every data reading step
    if np.any(delta_t == 0):
        raise ValueError("Timestamps not updated correctly")


def check_nan_values(data: dict) -> None:
    """
    Checks that each data dictionary in the dataset doesn't contain missing values.
    If the dictionary contains missing values, an error is raised.

    :param data:the data dictionary to analyse
    :raises ValueError: the dictionary contains invalid data
    """
    # consider empty string and numpy.inf as na values
    pd.set_option("mode.use_inf_as_na", True)
    # load data as pandas dataframe
    df = pd.DataFrame.from_dict(data)
    if df.isnull().values.any():
        raise ValueError("Missing values in the dictionary")


def get_demo_files(demo_folder_path: str) -> list[str]:
    """
    Retrieves all the demonstrations files in the .json format in the dataset
    folder.

    :param demo_folder_path: the path of the dataset
    :return: a list containing all file paths corresponding to demonstrations
    """
    files_paths = list(Path(demo_folder_path).rglob("*.json"))
    all_demonstration_files = [str(path) for path in files_paths]
    all_demonstration_files.sort()
    return all_demonstration_files
