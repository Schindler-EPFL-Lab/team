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
        cls.regression_j_space = np.array([[1, 2, 3, 4, 5, 6],
                                           [7, 8, 9, 10, 11, 12]])
        cls.translation_matrix = np.array([[1, 0, 0],
                                           [0, 1, 0],
                                           [1, 1, 1]])
        cls.translation_matrix_j_space = np.array([[1, 0, 0, 0, 0, 0],
                                                   [0, 1, 0, 0, 0, 0],
                                                   [0, 0, 1, 0, 0, 0],
                                                   [0, 0, 0, 1, 0, 0],
                                                   [0, 0, 0, 0, 1, 0],
                                                   [1, 1, 1, 1, 1, 1]])
        cls.rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)]
            ])
        cls.rotation_matrix_j_space = np.array([
            [np.cos(angle_radians), np.sin(angle_radians), 0, 0, 0, 0],
            [-np.sin(angle_radians), np.cos(angle_radians), 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, np.cos(angle_radians), np.sin(angle_radians), 0],
            [0, 0, 0, -np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 0, 0, 0, 1]
            ])
        cls.scaling_matrix = np.array([[scaling_factor, 0, 0],
                                       [0, scaling_factor, 0],
                                       [0, 0, scaling_factor]])
        cls.scaling_matrix_j_space = np.array([[scaling_factor, 0, 0, 0, 0, 0],
                                               [0, scaling_factor, 0, 0, 0, 0],
                                               [0, 0, scaling_factor, 0, 0, 0],
                                               [0, 0, 0, scaling_factor, 0, 0],
                                               [0, 0, 0, 0, scaling_factor, 0],
                                               [0, 0, 0, 0, 0, scaling_factor]])
        cls.dimensions = np.shape(cls.regression)[1]
        # Data for testing gmcc metric between different trajectories
        nb_samples = 100
        steps = np.linspace(0, nb_samples, num=nb_samples).reshape(nb_samples, 1)
        cls.diagonal = np.full_like(steps, steps, shape=(nb_samples, 3))
        cls.sinus = np.full_like(steps, np.sin(steps), shape=(nb_samples, 3))
        cls.diagonal_j = np.full_like(steps, steps, shape=(nb_samples, 6))
        cls.sinus_j = np.full_like(steps, np.sin(steps), shape=(nb_samples, 6))

    def test_gmcc_metric(self) -> None:
        gmcc_value = symmetric_gmcc(self.regression, self.regression)
        self.assertAlmostEqual(gmcc_value, 1, delta=0.02)

    def test_translation(self) -> None:
        reproduction = np.matmul(self.regression, self.translation_matrix)
        gmcc_value = symmetric_gmcc(self.regression, reproduction)
        self.assertAlmostEqual(gmcc_value, 1, delta=0.02)

    def test_rotation(self) -> None:
        reproduction = np.matmul(self.regression, self.rotation_matrix)
        gmcc_value = symmetric_gmcc(self.regression, reproduction)
        self.assertAlmostEqual(gmcc_value, 1, delta=0.02)

    def test_scaling(self) -> None:
        reproduction = np.matmul(self.regression, self.scaling_matrix)
        gmcc_value = symmetric_gmcc(self.regression, reproduction)
        self.assertAlmostEqual(gmcc_value, 1, delta=0.02)

    def test_difference(self) -> None:
        gmcc_value = symmetric_gmcc(self.diagonal, self.sinus)
        self.assertNotAlmostEqual(gmcc_value, 1, delta=0.1)

    def test_translation_j_space(self) -> None:
        reproduction = np.matmul(self.regression_j_space,
                                 self.translation_matrix_j_space)
        gmcc_value = symmetric_gmcc(self.regression_j_space, reproduction)
        self.assertAlmostEqual(gmcc_value, 1, delta=0.02)

    def test_rotation_j_space(self) -> None:
        reproduction = np.matmul(self.regression_j_space, self.rotation_matrix_j_space)
        gmcc_value = symmetric_gmcc(self.regression_j_space, reproduction)
        self.assertAlmostEqual(gmcc_value, 1, delta=0.02)

    def test_scaling_j_space(self) -> None:
        reproduction = np.matmul(self.regression_j_space, self.scaling_matrix_j_space)
        gmcc_value = symmetric_gmcc(self.regression_j_space, reproduction)
        self.assertAlmostEqual(gmcc_value, 1, delta=0.02)

    def test_difference_j_space(self) -> None:
        gmcc_value = symmetric_gmcc(self.diagonal_j, self.sinus_j)
        self.assertNotAlmostEqual(gmcc_value, 1, delta=0.1)
