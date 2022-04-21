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
        pe = ProbabilisticEncoding(
            data,
            max_nb_components=10,
            min_nb_components=2,
            iterations=1,
            random_state=0
        )
        # check best number GMM components
        self.assertEqual(pe.gmm.n_components, 3)
        # check norm of first covariance matrix
        self.assertEqual(int(np.linalg.norm(pe.gmm.covariances_[0])), 249)


if __name__ == "__main__":
    unittest.main(exit=False)
