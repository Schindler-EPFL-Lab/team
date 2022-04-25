import unittest
import os

import numpy as np

from arco.learning_from_demo.trajectories import Trajectories


class TrajectoriesTest(unittest.TestCase):

    @staticmethod
    def _get_file_path() -> str:

        # standard data file to perform tests
        filename = "manual_trajectory.json"
        file_dir = os.path.join(os.path.dirname(__file__), "trajectory_data")
        filename_path = os.path.join(file_dir, filename)
        return filename_path

    def test_load_single_trajectory(self):

        trajectory_path = self._get_file_path()
        trajectory = Trajectories.load_single_trajectory(trajectory_path)
        self.assertEqual(
            trajectory.joints_trajectories.all(),
            np.array(
                [
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                ]
            ).all(),
        )
        self.assertEqual(
            trajectory.timestamps.all(),
            np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]).all()
        )
