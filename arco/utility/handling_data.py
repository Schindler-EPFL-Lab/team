from typing import Optional

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


def read_json_file(filename_path):
    """
    Read json file in a pandas dataframe
    :param filename_path: filename path
    :return: the dataframe
    """
    df = pd.read_json(filename_path)
    return df
