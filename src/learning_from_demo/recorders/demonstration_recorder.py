import json
import os
from pathlib import Path

from learning_from_demo.utility.handling_data import (
    check_data_timestamps,
    check_nan_values,
    create_default_dict,
)
from learning_from_demo.utility.logger import log


class DemonstrationRecorder:
    """
    Records demonstrated trajectories in the json format.
    """

    def __init__(self, data_path: str) -> None:
        """
        Class constructor. Takes as argument the filepath to store the data.
        :param path_to_store_demo: filepath destination
        """

        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)

        self.dest_path = self.next_filename
        self.data = create_default_dict()

    @property
    def next_filename(self) -> str:
        # Count files in destination folder
        nb_demonstrations = len(list(Path(self.data_path).rglob("*.json")))
        # Define file name
        filename = f"demonstration_{nb_demonstrations + 1}.json"
        return str(Path(self.data_path).joinpath(filename))

    def create_file(self) -> None:
        """
        Writes the recorded data to a new file.

        If the file already exists, it is not
        allowed to overwrite it and it prints an error log to the console.
        Before saving the file, content checks verify its integrity.
        The demonstration record will point to a new file for future recordings.
        The program is not stopped and the robot shutdown operations can be executed.
        """
        try:
            assert not os.path.isfile(
                self.dest_path
            ), "File already exists, not allowed to overwrite it"
            check_data_timestamps(self.data)
            check_nan_values(self.data)
            with open(self.dest_path, "w") as file:
                file.write(json.dumps(self.data))
            self.dest_path = self.next_filename
        except AssertionError as e:
            log.error(e)

    def update(self, tmp_dict: dict) -> None:
        """
        Updates the data structure with the incoming flow of information.
        :param tmp_dict: robot end effector, joints and timestamp status
        """
        for key in self.data.keys():
            self.data[key].append(tmp_dict[key])
