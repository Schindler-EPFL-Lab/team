import pathlib
import unittest

import numpy as np

from arco.learning_from_demo.probabilistic_encoding import ProbabilisticEncoding
from arco.learning_from_demo.data_preprocessing import DataPreprocessing


class ProbabilisticEncodingTest(unittest.TestCase):

    def test_probabilistic_encoding(self):

        sampling_rate = 100
        base_path = pathlib.Path(__file__).parent.absolute()
        data_path = str(pathlib.Path(base_path, "data"))
        dp = DataPreprocessing(data_path, sampling_rate=sampling_rate)
        dp.preprocessing()
        data = np.array(dp.aligned_and_padded_trajectories)
        pe = ProbabilisticEncoding(data, max_k=10)
        # check number of demo
        self.assertEqual(pe.nb_demo, 2)
        # check demonstrations length
        self.assertEqual(pe.length_demo, 1027)
        # check number features
        self.assertEqual(pe.nb_features, 7)
        # check best number GMM components
        self.assertEqual(pe.best_gmm.n_components, 5)
        # check best covariance type
        self.assertEqual(pe.best_gmm.covariance_type, 'diag')


if __name__ == "__main__":
    unittest.main(exit=False)
