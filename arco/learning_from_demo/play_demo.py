import os

import pandas as pd
import numpy as np

from rws2.RWS_wrapper import RwsWrapper


class PlayBack:
    """
    Plays demonstrated trajectories saved in the json format.

    See 'recording.py' for the recording procedure.
    """

    def __init__(self, filename_path: str, base_url: str) -> None:
        """
        Class constructor. It calls a method to read and split the data into timestamps
        and data required to reproduce the robot trajectory.
        :param filename_path: path to the json filename to read
        :param base_url: url address to establish communication with
        """
        # Load data as pandas dataframe
        self.df = None
        self.timestamps = None
        self.data = None
        self.target_generator = None
        self.current_pose = None
        self.next_target = None
        self.iter = None
        # control the smoothness of the reproduction
        self.tol_pos = 100
        self.tol_ori = 1
        self.rws = RwsWrapper(robot_url=base_url)
        self.read_split_data(filename_path=filename_path)

    def read_split_data(self, filename_path: str) -> None:
        """
        The method reads the .json file path passed as argument into a pandas dataframe.
        The timestamp column is separated from the information required to play the
        demonstration.
        :param filename_path: path to the json file to read
        """
        # check that the file format is .json
        assert os.path.splitext(filename_path)[1] == ".json", "Erroneous file format." \
                                                              ".json format required " \
                                                              "for reading the file"
        self.df = pd.read_json(filename_path)
        self.timestamps = self.df["timestamp"]
        # consider only tcp_pos, tcp_ori and robot configuration
        self.data = self.df.iloc[:, 1:12]
        # iterator over the dataframe rows
        self.target_generator = self.data.iterrows()

    def get_next_target(self) -> None:
        """
        Gets the next row of the dataframe as the next target.
        """
        self.iter, self.next_target = next(self.target_generator)

    def is_target_eligible(self) -> bool:
        """
        Checks that the distance between the actual robot pose and the new target is
        large enough. It avoids to perform micro movements.
        :return: boolean assessing if the target needs to be considered or not
        """
        is_pos_different = self.compute_pos_distance() > self.tol_pos
        is_ori_different = self.compute_ori_distance() > self.tol_ori
        return is_pos_different or is_ori_different

    def compute_pos_distance(self) -> float:
        """
        Computes the Euclidean norm between current pose and next target pose.
        :return: the distance between the current pose and the next target pose in mm
        """
        return np.sqrt(np.sum(np.power(self.next_target[:3] - self.current_pose[:3], 2))
                       )

    def compute_ori_distance(self) -> float:
        """
        Very naive way to compute the difference in orientation
        :return:
        """
        # TODO find a better expression
        return np.sqrt(np.sum(np.power(self.next_target[3:7] - self.current_pose[3:7],
                                       2)))

    def play(self) -> None:
        """
        Firstly, it sends the robot to the initial trajectory pose. Then, if the new
        target is far enough, it executes the motion, otherwise it queries the next one.
        It repeats until the end of the demonstrated trajectory and finally motors are
        turned off.
        """
        self.rws.set_RAPID_variable("program_running", "TRUE")
        for t in range(len(self.timestamps)):
            self.get_next_target()
            if t == 0:
                self.current_pose = self.next_target
                self.execute_target(self.set_target())
            else:
                if self.is_target_eligible():
                    self.execute_target(self.set_target())
                    self.current_pose = self.next_target
                elif self.iter == len(self.timestamps) - 1:
                    break
                else:
                    pass
        self.rws.robot.motors_off()

    def set_target(self) -> str:
        """
        Creates the target in the required format accepted by RAPID.
        :return: string of a list of lists containing the tcp_pos, tcp_ori, robot config
                 and the external axis
        """
        pos_list = self.next_target.values[0:3].tolist()
        ori_list = self.next_target.values[3:7].tolist()
        config_list = self.next_target.values[7:11].tolist()
        ext_axis_list = [9e9, 9e9, 9e9, 9e9, 9e9, 9e9]
        target = str([pos_list, ori_list, config_list, ext_axis_list])
        return target

    def execute_target(self, target):
        """
        Manipulates the RAPID variable to set new target and reach it.
        :param target: new target value to assign to the robot
        """
        self.rws.set_RAPID_variable("Loc", target)
        self.rws.complete_instruction()
