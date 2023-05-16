import unittest

import numpy as np
from spatialmath import SE3

from team.gofa_trajectory import GoFaTrajectory


class GoFaTrajectoriesTest(unittest.TestCase):
    def test_no_movement(self) -> None:
        trajectory = GoFaTrajectory(
            np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]),
        )

        trajectory._translations = [
            SE3(1, 2, 3),
            SE3(0, 0, 0),
            SE3(0, 0, 0),
            SE3(0, 0, 0),
            SE3(0, 0, 0),
            SE3(0, 0, 0),
        ]

        trajectory.tcp = trajectory._compute_tcp()
        for tcp in trajectory.tcp:
            np.testing.assert_array_almost_equal(tcp, np.array([1, 2, 3]))

        trajectory._translations = [
            SE3(1, 2, 3),
            SE3(1, 2, 3),
            SE3(0, 0, 0),
            SE3(0, 0, 0),
            SE3(0, 0, 0),
            SE3(0, 0, 0),
        ]

        trajectory.tcp = trajectory._compute_tcp()

        for tcp in trajectory.tcp:
            np.testing.assert_array_almost_equal(tcp, np.array([2, 4, 6]))

    def test_90_deg_rot(self):
        trajectory = GoFaTrajectory(
            np.array([[0, 0, 0, 0, 0, 0], [22 / 7, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        )
        trajectory._translations = [
            SE3(1, 0, 0),
            SE3(0, 0, 0),
            SE3(0, 0, 0),
            SE3(0, 0, 0),
            SE3(0, 0, 0),
            SE3(0, 0, 0),
        ]
        trajectory.tcp = trajectory._compute_tcp()

        np.testing.assert_array_almost_equal(trajectory.tcp[0], np.array([1, 0, 0]))
        np.testing.assert_array_almost_equal(
            trajectory.tcp[1], np.array([-1, 0, 0]), decimal=3
        )
        np.testing.assert_array_almost_equal(trajectory.tcp[2], np.array([1, 0, 0]))
