import json
import unittest
import os
import shutil
from pathlib import Path

import numpy as np

from learning_from_demo.dynamical_movement_primitives import DynamicMovementPrimitives


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
        self.assertEqual(dmp._tau, 7.17)

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

    def test_forcing_term_weights(self):

        dmp = self._create_dmp()
        np.testing.assert_almost_equal(
            dmp._w,
            np.array(
                [
                    [
                        474.96453587,
                        469.02587747,
                        463.83989184,
                        458.51502333,
                        442.62147762,
                        396.48734359,
                        320.01913768,
                        271.04395375,
                        289.29138663,
                        293.74963017,
                        231.83046191,
                        139.08995199,
                        27.99125372,
                        -135.29381345,
                        -400.82430753,
                        -821.52764625,
                        -1453.51154543,
                        -2346.81382914,
                        -3496.94949527,
                        -4708.38117518,
                        -5333.02209821,
                        -4088.81059042,
                        188.03745328,
                        6284.72090483,
                        10011.81374101,
                        8844.43796127,
                        5003.57936567,
                        1691.54708522,
                        -40.99629019,
                        -549.78442906,
                    ],
                    [
                        863.83200902,
                        886.51910628,
                        920.65572114,
                        992.39696417,
                        1105.62342328,
                        1216.1452978,
                        1194.2370312,
                        932.16532997,
                        625.09925435,
                        541.61444827,
                        607.59667306,
                        676.94034169,
                        696.36761015,
                        621.51904075,
                        391.33979404,
                        -68.68119193,
                        -854.44387498,
                        -2084.97323962,
                        -3860.28503149,
                        -6099.04570439,
                        -8146.33117401,
                        -8277.04571889,
                        -4197.79546375,
                        3920.41873967,
                        11033.65363055,
                        12196.84864443,
                        8586.94942533,
                        4330.46315186,
                        1564.30425903,
                        384.41361405,
                    ],
                    [
                        -787.54219472,
                        -802.07734362,
                        -817.90664041,
                        -853.05262615,
                        -912.41840377,
                        -983.26976718,
                        -1028.79633593,
                        -1007.42256821,
                        -939.64193606,
                        -883.90371888,
                        -850.48845573,
                        -830.29820204,
                        -804.48580025,
                        -723.93710906,
                        -518.26146768,
                        -104.13035099,
                        621.80920896,
                        1787.19200437,
                        3509.11732362,
                        5744.47677381,
                        7909.24689689,
                        8358.42440277,
                        4769.4725168,
                        -2967.62425748,
                        -10155.49455477,
                        -11791.39772961,
                        -8659.15869933,
                        -4657.93516535,
                        -1958.54057525,
                        -756.56553491,
                    ],
                    [
                        668.82412941,
                        652.46797647,
                        641.4687002,
                        632.1459437,
                        611.60326016,
                        559.46077969,
                        486.06316111,
                        464.13592818,
                        511.7196357,
                        505.73480035,
                        402.70584453,
                        268.38597357,
                        120.20665295,
                        -90.95822408,
                        -436.67988788,
                        -992.21131221,
                        -1836.75546406,
                        -3041.3228658,
                        -4603.00472416,
                        -6257.40657091,
                        -7115.88547528,
                        -5405.21274039,
                        512.33852244,
                        9018.01648769,
                        14345.35521405,
                        12944.93117328,
                        7814.53839883,
                        3368.49219344,
                        1078.46839285,
                        454.85705166,
                    ],
                    [
                        -153.89647354,
                        -170.48755116,
                        -199.12401674,
                        -254.72444697,
                        -339.07970117,
                        -412.32360243,
                        -345.41811963,
                        -32.11318767,
                        302.89007796,
                        357.64746827,
                        230.71051147,
                        110.48580994,
                        48.92508078,
                        44.05689398,
                        89.02006723,
                        179.11817507,
                        313.00739309,
                        488.91871173,
                        690.67072069,
                        855.74413482,
                        825.10677248,
                        334.18586256,
                        -773.98179895,
                        -2082.64223435,
                        -2660.51753498,
                        -2163.65908863,
                        -1255.70226141,
                        -615.13451002,
                        -371.47371241,
                        -375.11480709,
                    ],
                    [
                        -481.74074884,
                        -474.32978138,
                        -470.32012261,
                        -469.60775643,
                        -464.12094845,
                        -437.58907067,
                        -388.59030772,
                        -359.6714057,
                        -377.16739914,
                        -374.18327912,
                        -311.41655261,
                        -220.9546995,
                        -113.06473968,
                        44.63527988,
                        302.02739582,
                        713.9731066,
                        1339.85868796,
                        2234.23088288,
                        3398.49260161,
                        4642.68013128,
                        5315.95165271,
                        4106.74988868,
                        -217.2816049,
                        -6500.72096316,
                        -10494.91172514,
                        -9529.59259508,
                        -5770.30508545,
                        -2477.62277355,
                        -761.71465009,
                        -278.51932126,
                    ],
                ]
            ),
            decimal=3,
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
            dmp.y[-1, :, 0], np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]), decimal=2,
        )

    def test_optimal_paramaters(self):

        # check the dmp parameters optimized to track a line over a small search space
        regression = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0.01, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                [0.02, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.03, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
                [0.04, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            ]
        )
        dmp = DynamicMovementPrimitives(
            regression=regression,
            c_order=1,
            goal_joints=regression[-1, 1:],
            initial_joints=regression[0, 1:],
            search_space=[(2, 4), (2, 4)],
        )
        dmp.compute_joint_dynamics(goal=regression[-1, 1:], y_init=regression[0, 1:])
        np.testing.assert_equal(dmp._alpha_z, 3 * np.ones(dmp._nb_joints))
        self.assertEqual(dmp._n_rfs, 2)

    def test_parameters_loading(self):

        # check that the dmp parameters are loaded correctly
        path = Path(os.path.join(os.path.dirname(__file__), "dmp_data"))
        dmp = DynamicMovementPrimitives.load_dmp(
            dir_path=path, g_joints=np.ones(6), i_joints=np.ones(6)
        )
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
