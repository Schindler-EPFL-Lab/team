import os

import json

from arco.utility.logger import log
from arco.utility.handling_data import create_default_dict, check_reading_files, \
    check_nan_values, check_data_timestamps


class DemonstrationRecorder:
    """
    Records demonstrated trajectories in the json format.
    """
    def __init__(self, path_to_store_demo: str) -> None:
        """
        Class constructor. Takes as argument the filepath to store the data.
        :param path_to_store_demo: filepath destination
        """
        self.dest_path = path_to_store_demo
        self.data = create_default_dict()

    def create_file(self) -> None:
        """
        Writes the recorded data to a new file. If the file already exists, it is not
        allowed to overwrite it and it prints an error log to the console.
        Before saving the file, content checks verify its integrity.
        The program is not stopped and the robot shutdown operations can be executed.
        """
        try:
            assert not os.path.isfile(
                self.dest_path
            ), "File already exists, not allowed to overwrite it"
            check_data_timestamps(self.data)
            check_nan_values(self.data)
            check_reading_files(self.data)
            with open(self.dest_path, "w") as file:
                file.write(json.dumps(self.data))
        except AssertionError as e:
            log.error(e)

    def update(self, tmp_dict: dict) -> None:
        """
        Updates the data structure with the incoming flow of information.
        :param tmp_dict: robot end effector, joints and timestamp status
        """
        for key in self.data.keys():
            self.data[key].append(tmp_dict[key])
