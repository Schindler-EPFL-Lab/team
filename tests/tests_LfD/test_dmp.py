import json
import os
import shutil
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeResult

from team.dynamical_movement_primitives import DynamicMovementPrimitives


class DynamicalMovementPrimitivesTest(unittest.TestCase):
    @staticmethod
    def _create_dmp():

        # standard data file to perform tests
        filename = "regression.npy"
        file_dir = os.path.join(os.path.dirname(__file__), "dmp_data")
        filename_path = os.path.join(file_dir, filename)
        regression = np.load(filename_path)
        dmp = DynamicMovementPrimitives(
            regression=regression,
            c_order=1,
            goal_joints=regression[0, 1:],
            initial_joints=regression[-1, 1:],
        )
        dmp.set_alpha_z_and_n_rfs(alpha_z=18 * np.array([1, 1, 1, 1, 1, 1]), n_rfs=30)
        return dmp

    def test_dmp_time_constant(self):
        dmp = self._create_dmp()
        self.assertEqual(dmp._tau, 7.21)

    def test_rbf_kernels(self):

        dmp = self._create_dmp()
        # test rbf kernel centers
        np.testing.assert_almost_equal(
            dmp._c[:, 0],
            np.array(
                [
                    1.0,
                    0.96073708,
                    0.87123992,
                    0.76110906,
                    0.64772577,
                    0.54066529,
                    0.44462494,
                    0.36134538,
                    0.29085398,
                    0.23225511,
                    0.18422073,
                    0.1452849,
                    0.11401186,
                    0.08908432,
                    0.06934243,
                    0.05379354,
                    0.04160556,
                    0.03209191,
                    0.02469315,
                    0.01895798,
                    0.01452538,
                    0.01110853,
                    0.00848096,
                    0.00646472,
                    0.00492066,
                    0.00374033,
                    0.00283956,
                    0.00215319,
                    0.00163094,
                    0.0012341,
                ]
            ),
            decimal=3,
        )
        # test rbf kernels covariances
        np.testing.assert_almost_equal(
            dmp._D[:, 0],
            np.array(
                [
                    2.14441789e03,
                    4.12720597e02,
                    2.72556489e02,
                    2.57144065e02,
                    2.88413967e02,
                    3.58399268e02,
                    4.76648026e02,
                    6.65276700e02,
                    9.62711440e02,
                    1.43274969e03,
                    2.18059799e03,
                    3.38013810e03,
                    5.32005150e03,
                    8.48198009e03,
                    1.36733865e04,
                    2.22541407e04,
                    3.65241900e04,
                    6.03888400e04,
                    1.00503699e05,
                    1.68250405e05,
                    2.83154544e05,
                    4.78812503e05,
                    8.13191896e05,
                    1.38657058e06,
                    2.37284098e06,
                    4.07423991e06,
                    7.01720367e06,
                    1.21205333e07,
                    2.09908897e07,
                    2.09908897e07,
                ]
            ),
            decimal=1,
        )

    def test_dmp_convergence(self):

        dmp = self._create_dmp()
        _ = dmp.compute_joint_dynamics(
            goal=np.array([20.874, 23.377, 25.108, 28.372, -53.129, -18.937]),
            y_init=np.array([-3.002, -20.142, 66.025, -4.175, -46.443, 4.869]),
        )
        # test that the last combination of joint angles corresponds to the target one
        np.testing.assert_almost_equal(
            dmp.y[-1, :, 0],
            np.array([20.874, 23.377, 25.108, 28.372, -53.129, -18.937]),
            decimal=1,
        )

    def test_convergence_to_dummy_data(self):

        # manually generated regression line to track, the method coefficients have been
        # tuned accordingly. If test fails, check that the method has not been changed
        dmp = DynamicMovementPrimitives(
            regression=np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0.01, 0, 0, 0, 0, 0, 0],
                    [0.02, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.03, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                ]
            ),
            c_order=1,
            goal_joints=np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
            initial_joints=np.array([0, 0, 0, 0, 0, 0]),
        )
        dmp.set_alpha_z_and_n_rfs(alpha_z=1.8 * np.array([1, 1, 1, 1, 1, 1]), n_rfs=2)
        _ = dmp.compute_joint_dynamics(
            goal=np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
            y_init=np.array([0, 0, 0, 0, 0, 0]),
        )
        # test that the last combination of joint angles corresponds to the target one
        np.testing.assert_almost_equal(
            dmp.y[-1, :, 0],
            np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
            decimal=2,
        )

    def test_parameters_loading(self):

        # check that the dmp parameters are loaded correctly
        path = Path(os.path.join(os.path.dirname(__file__), "dmp_data"))
        dmp = DynamicMovementPrimitives.load_dmp(dir_path=path)
        alpha_z = dmp._alpha_z
        n_rfs = dmp._n_rfs
        np.testing.assert_equal(alpha_z, 12 * np.ones(dmp._nb_joints))
        self.assertEqual(n_rfs, 180)

    def test_information_saving(self):

        dmp = self._create_dmp()
        # specify the dmp parameters to save
        dmp.set_alpha_z_and_n_rfs(alpha_z=np.ones([dmp._nb_joints]), n_rfs=10)
        # specify the regression function to save
        dmp.regression = np.array(
            [[0, 1, 1, 1, 1, 1, 1], [0.01, 2, 2, 2, 2, 2, 2], [0.02, 3, 3, 3, 3, 3, 3]]
        )
        store_path = Path(os.path.join(os.path.dirname(__file__), "saved_data"))
        dmp.save_dmp(dir_path=store_path)
        with open(store_path.joinpath("dmp_parameters.json"), "r") as f:
            data = json.load(f)
        # check that the dmp parameters have been saved correctly
        np.testing.assert_equal(data["alpha_z"], np.ones([dmp._nb_joints]))
        self.assertEqual(data["n_rfs"], 10)
        # check that regression function has been saved correctly
        regression_path = store_path.joinpath("regression.npy")
        data = np.load(str(regression_path))
        np.testing.assert_equal(
            data,
            np.array(
                [
                    [0, 1, 1, 1, 1, 1, 1],
                    [0.01, 2, 2, 2, 2, 2, 2],
                    [0.02, 3, 3, 3, 3, 3, 3],
                ]
            ),
        )
        shutil.rmtree(store_path)

    def test_optimizer_stopping_criterion(self):

        dmp = self._create_dmp()
        res = OptimizeResult()
        res.x_iters = [[3, 2], [1, 2], [1, 3], [1, 2], [1, 2]]
        self.assertTrue(dmp._stop_at_convergence(res))
