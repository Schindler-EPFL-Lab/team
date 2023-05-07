import unittest

import numpy as np

from team.utility.accuracy_metric import symmetric_gmcc


class GmccTest(unittest.TestCase):
    @classmethod
    def setUp(cls) -> None:
        rotation_angle = 90
        angle_radians = rotation_angle*(np.pi/180)
        scaling_factor = 2
        cls.regression = np.array([[1, 2, 3],
                                   [7, 8, 9]])
        cls.translation_matrix = np.array([[1, 0, 1],
                                           [0, 1, 1],
                                           [0, 0, 1]])
        cls.rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)]
            ])
        cls.scaling_matrix = np.array([[scaling_factor, 0, 0],
                                       [0, scaling_factor, 0],
                                       [0, 0, scaling_factor]])
        cls.dimensions = np.shape(cls.regression)[1]

    def test_gmcc_metric(self) -> None:
        gmcc_value = symmetric_gmcc(self.regression, self.regression)
        self.assertAlmostEqual(gmcc_value, 1, places=1)

    def test_translation(self) -> None:
        reproduction = np.matmul(self.regression, self.translation_matrix)
        gmcc_value = symmetric_gmcc(self.regression, reproduction)
        self.assertAlmostEqual(gmcc_value, 1, places=1)

    def test_rotation(self) -> None:
        reproduction = np.matmul(self.regression, self.rotation_matrix)
        gmcc_value = symmetric_gmcc(self.regression, reproduction)
        self.assertAlmostEqual(gmcc_value, 1, places=1)

    def test_scaling(self) -> None:
        reproduction = np.matmul(self.regression, self.scaling_matrix)
        gmcc_value = symmetric_gmcc(self.regression, reproduction)
        self.assertAlmostEqual(gmcc_value, 1, places=1)
