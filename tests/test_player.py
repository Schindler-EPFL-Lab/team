import os
import unittest
from unittest import mock

import pandas as pd

from arco.learning_from_demo.demonstration_player import DemonstrationPlayer
from arco.utility.handling_data import create_default_dict


# unittest will test all the methods whose name starts with 'test'
class PlayerTest(unittest.TestCase):

    # return True or False
    def test_default_dictionary_init(self) -> bool:
        is_well_init = True
        default_dict = create_default_dict()
        for key in default_dict.keys():
            if not self.assertEqual(default_dict[key], []):
                is_well_init = False
        return is_well_init

    def test_reading_file(self):
        # standard data file to perform tests
        filename = "test_data.json"
        file_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "demonstrations"
        )
        filename_path = os.path.join(file_dir, filename)
        extension = os.path.splitext(filename_path)[1]
        # checks that the file exists, inside a valid directory and right file extension
        self.assertTrue(os.path.exists(filename_path))
        self.assertTrue(os.path.isdir(file_dir))
        self.assertEqual(extension, '.json')
        dataframe = pd.read_json(filename_path)
        default_dict = create_default_dict()
        # checks that the dataframe has the same keys as the recorded dictionary
        for df_key, dict_key in zip(dataframe.keys(), default_dict.keys()):
            self.assertEqual(df_key, dict_key)

    @mock.patch('rws2.RWS_wrapper')
    def test_demonstration_length(self, mock_rws):
        url = "https://localhost:8881"
        # standard data file to perform tests
        filename = "test_data.json"
        file_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "demonstrations"
        )
        filename_path = os.path.join(file_dir, filename)
        play = DemonstrationPlayer(filename_path=filename_path, base_url=url)
        play.rws = mock_rws
        # before playing the demonstration, iteration should be 0
        self.assertEqual(play.iter, 0)
        play.play()
        # after the end of the demonstration, iteration should equal the number of steps
        self.assertEqual(play.iter, len(play.timestamps) - 1)

    @mock.patch('rws2.RWS_wrapper')
    def test_target_integrity(self, mock_rws):
        url = "https://localhost:8881"
        # standard data file to perform tests
        filename = "test_data.json"
        file_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "demonstrations"
        )
        filename_path = os.path.join(file_dir, filename)
        play = DemonstrationPlayer(filename_path=filename_path, base_url=url)
        play.rws = mock_rws
        # checks that no value is None and the type is int or float
        for t in range(len(play.timestamps) - 1):
            play.get_next_target()
            for value in play.next_target:
                self.assertTrue(value is not None)
                self.assertTrue(isinstance(value, int) or isinstance(value, float))
            target = play.set_target()
            # checks that the target to pass is a string as rws requires
            self.assertTrue(isinstance(target, str))

    @mock.patch('rws2.RWS_wrapper')
    def test_positive_error(self, mock_rws):
        url = "https://localhost:8881"
        # standard data file to perform tests
        filename = "test_data.json"
        file_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "demonstrations"
        )
        filename_path = os.path.join(file_dir, filename)
        play = DemonstrationPlayer(filename_path=filename_path, base_url=url)
        play.rws = mock_rws
        # checks that the error between a couple of consecutive targets is always >= 0
        play.get_next_target()
        play.current_pose = play.next_target
        for t in range(len(play.timestamps) - 1):
            play.get_next_target()
            error = play.compute_difference()
            self.assertTrue(error >= 0)
            play.current_pose = play.next_target


if __name__ == "__main__":
    unittest.main(exit=False)
