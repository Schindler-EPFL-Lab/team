import pathlib
import unittest

import numpy as np

from learning_from_demo.probabilistic_encoding import ProbabilisticEncoding
from learning_from_demo.aligned_trajectories import AlignedTrajectories
from learning_from_demo.gaussian_mixture_regression import GMR


class ProbabilisticEncodingTest(unittest.TestCase):
    @staticmethod
    def _create_trajectory_and_prob_encoding() -> tuple[
        AlignedTrajectories, ProbabilisticEncoding
    ]:
        base_path = pathlib.Path(__file__).parent.absolute()
        data_path = str(pathlib.Path(base_path, "data"))
        trajectories = AlignedTrajectories.load_dataset_and_preprocess(data_path)
        pe = ProbabilisticEncoding(
            trajectories,
            max_nb_components=10,
            min_nb_components=2,
            iterations=1,
            random_state=0,
        )
        return trajectories, pe

    def test_probabilistic_encoding(self):

        _, pe = self._create_trajectory_and_prob_encoding()
        # check best number GMM components
        self.assertEqual(pe.gmm.n_components, 5)
        # check norm of covariance matrices
        for i, norm in enumerate([217, 80, 50, 281, 58]):
            self.assertEqual(int(np.linalg.norm(pe.gmm.covariances_[i])), norm)

    def test_gmr_implementation(self):

        traj, pe = self._create_trajectory_and_prob_encoding()
        # compute regression curve
        regression = GMR(traj, pe)
        # check prediction vector, first timestamp
        np.testing.assert_array_almost_equal(
            regression.prediction[0, :],
            np.array([0.0, 0.43, -21.18, 68.83, 6.32, -49.99, -11.77]),
            decimal=1,
        )

        # check prediction vector, last timestamp
        np.testing.assert_array_almost_equal(
            regression.prediction[-1, :],
            np.array([12.83, 19.76, 43.12, -5.45, 26.73, -44.88, -24.18]),
            decimal=1,
        )
