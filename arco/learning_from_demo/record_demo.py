import os

import json

from arco.utility.logger import log


class RecordDemo:
    """
    Records demonstrated trajectories in the json format.
    """
    def __init__(self, path_to_store_demo: str) -> None:
        """
        Class constructor. Takes as argument the filepath to store the data.
        :param path_to_store_demo: filepath destination
        """
        self.dest_path = path_to_store_demo
        self.data = DataStructure()

    def create_file(self) -> None:
        """
        Writes the recorded data to a new file. If the file already exists, it is not
        allowed to overwrite it and it prints an error log to the console. The program
        is not stopped and the robot shutdown operations can be executed.
        """
        try:
            assert not os.path.isfile(
                self.dest_path
            ), "File already exists, not allowed to overwrite it"
            with open(self.dest_path, "w") as file:
                file.write(json.dumps(self.data.data))
        except AssertionError as e:
            log.error(e)
            pass

    def update(self, tmp_dict: dict) -> None:
        """
        Updates the data structure with the incoming flow of information.
        :param tmp_dict: robot end effector, joints and timestamp status
        """
        for key in self.data.keys:
            self.data.data[key].append(tmp_dict[key])


class DataStructure:
    def __init__(self) -> None:
        """
        Class constructor. Defines the dictionary data structure to be recorded.
        """
        self.keys = [
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
        self.data = {key: [] for key in self.keys}
