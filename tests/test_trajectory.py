import unittest
import os

import numpy as np

from learning_from_demo.aligned_trajectories import AlignedTrajectories
from learning_from_demo.trajectory import Trajectory


class TrajectoriesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # standard data file to perform tests
        filename = "manual_trajectory.json"
        file_dir = os.path.join(os.path.dirname(__file__), "trajectory_data")
        filename_path = os.path.join(file_dir, filename)
        cls.trajectory = Trajectory.from_file(filename_path)

    def test_load_single_trajectory(self) -> None:
        np.testing.assert_array_almost_equal(
            self.trajectory.joints,
            np.array(
                [
                    [110, 120, 130, 140, 150, 160],
                    [111, 121, 131, 141, 151, 161],
                    [112, 122, 132, 142, 152, 162],
                    [113, 123, 133, 143, 153, 163],
                    [114, 124, 134, 144, 154, 164],
                    [115, 125, 135, 145, 155, 165],
                    [116, 126, 136, 146, 156, 166],
                    [117, 127, 137, 147, 157, 167],
                    [118, 128, 138, 148, 158, 168],
                    [119, 129, 139, 149, 159, 169],
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            self.trajectory.timestamps,
            np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]),
        )

    def test_read_tcp(self) -> None:
        np.testing.assert_array_almost_equal(
            self.trajectory.tcp,
            np.array(
                [
                    [0, 10, 20],
                    [1, 11, 21],
                    [2, 12, 22],
                    [3, 13, 23],
                    [4, 14, 24],
                    [5, 15, 25],
                    [6, 16, 26],
                    [7, 17, 27],
                    [8, 18, 28],
                    [9, 19, 29],
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            self.trajectory.timestamps,
            np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]),
        )

    def test_padding(self) -> None:
        traj = Trajectory(
            np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        )
        traj.pad_end(10)
        self.assertEqual(len(traj), 10)

    def test_upsample(self) -> None:
        traj = Trajectory(
            np.array(
                [
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                ]
            )
        )
        traj.upsample(2)
        np.testing.assert_array_almost_equal(
            traj.get_joint(0), np.array([1, 1.5, 2, 2.5, 3])
        )

    def test_load_list_of_trajectories(self) -> None:
        # standard data file to perform tests
        list_traj = [
            Trajectory(
                np.array(
                    [
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        [1, 2, 4, 5, 6, 7, 8, 9, 10, 11],
                        [2, 3, 5, 7, 9, 11, 13, 15, 17, 19],
                    ]
                )
            ),
            Trajectory(
                np.array(
                    [
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        [1, 2, 4, 5, 6, 7, 8, 9, 10, 11],
                        [2, 3, 5, 7, 9, 11, 13, 15, 17, 19],
                    ]
                )
            ),
        ]
        trajectories = AlignedTrajectories.from_list_trajectories(list_traj)
        self.assertEqual(np.shape(trajectories.aligned_trajectories)[0], 2)
        self.assertEqual(np.shape(trajectories.aligned_trajectories)[1], 3)
        self.assertEqual(np.shape(trajectories.aligned_trajectories)[2], 7)

    def test_joints_to_string(self):
        # test that the string to store has been generated well
        self.assertEqual(
            self.trajectory.joints_to_string(),
            "[110.0, 120.0, 130.0, 140.0, 150.0, 160.0]\n"
            "[111.0, 121.0, 131.0, 141.0, 151.0, 161.0]\n"
            "[112.0, 122.0, 132.0, 142.0, 152.0, 162.0]\n"
            "[113.0, 123.0, 133.0, 143.0, 153.0, 163.0]\n"
            "[114.0, 124.0, 134.0, 144.0, 154.0, 164.0]\n"
            "[115.0, 125.0, 135.0, 145.0, 155.0, 165.0]\n"
            "[116.0, 126.0, 136.0, 146.0, 156.0, 166.0]\n"
            "[117.0, 127.0, 137.0, 147.0, 157.0, 167.0]\n"
            "[118.0, 128.0, 138.0, 148.0, 158.0, 168.0]\n"
            "[119.0, 129.0, 139.0, 149.0, 159.0, 169.0]\n"
            "[119.0, 129.0, 139.0, 149.0, 159.0, 169.0]",
        )
