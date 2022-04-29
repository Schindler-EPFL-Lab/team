import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from arco.learning_from_demo.demonstration_player import DemonstrationPlayer
from arco.learning_from_demo.trajectory import Trajectory
from arco.utility.handling_data import (
    create_default_dict,
    read_json_file,
    target_encoding,
)


class PlayerTest(unittest.TestCase):
    @staticmethod
    def _setup():
        return (
            PlayerTest._create_player(),
            Trajectory.from_file(PlayerTest._get_file_path()),
        )

    @staticmethod
    def _create_player() -> DemonstrationPlayer:
        url = "https://localhost:8881"
        player = DemonstrationPlayer(base_url=url)
        return player

    @staticmethod
    def _get_file_path() -> str:
        # standard data file to perform tests
        filename = "manual_trajectory.json"
        file_dir = os.path.join(os.path.dirname(__file__), "trajectory_data")
        filename_path = os.path.join(file_dir, filename)
        return filename_path

    def test_default_dictionary_init(self) -> bool:
        is_well_init = True
        default_dict = create_default_dict()
        for key in default_dict.keys():
            if not self.assertEqual(default_dict[key], []):
                is_well_init = False
        return is_well_init

    def test_reading_file(self):
        filename = "manual_trajectory.json"
        file_dir = os.path.join(os.path.dirname(__file__), "trajectory_data")
        filename_path = os.path.join(file_dir, filename)
        # checks that the file exists, inside a valid directory and right file extension
        self.assertTrue(os.path.exists(filename_path))
        self.assertTrue(os.path.isdir(file_dir))
        dataframe = pd.read_json(filename_path)
        default_dict = create_default_dict()
        # checks that the dataframe has the same keys as the recorded dictionary
        for df_key, dict_key in zip(dataframe.keys(), default_dict.keys()):
            self.assertEqual(df_key, dict_key)

    @mock.patch("rws2.RWS_wrapper")
    def test_target_integrity(self, mock_rws):
        play, trajectory = PlayerTest._setup()
        play.rws = mock_rws
        dataframe_test = read_json_file(self._get_file_path())
        # checks that all the read data are the expected ones
        for t, joints in enumerate(trajectory.joints):
            play.next_target = np.around(joints, decimals=6)
            self.assertEqual(
                play.next_target.all(),
                dataframe_test.iloc[t, -6:].to_numpy().round(decimals=6).all(),
            )
            target = play.set_target()
            # checks that the target to pass is exactly the string as rws requires
            self.assertEqual(
                target, target_encoding(dataframe_test.round(decimals=6), t)
            )

    @mock.patch("rws2.RWS_wrapper")
    def test_positive_error(self, mock_rws):
        play, trajectory = PlayerTest._setup()
        play.rws = mock_rws
        errors = [
            0,
            2.4494,
            2.4494,
            2.4494,
            2.4494,
            2.4494,
            2.4494,
            2.4494,
            2.4494,
            2.4494,
        ]
        # checks that the error between a couple of consecutive targets is correct
        play.next_target = trajectory.joints[0]
        play.current_pose = play.next_target
        for i, joints in enumerate(trajectory.joints):
            play.next_target = joints
            error = play.compute_difference()
            self.assertAlmostEqual(error, errors[i], places=3)
            play.current_pose = play.next_target
