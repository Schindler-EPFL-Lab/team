import pathlib
import unittest

import numpy as np

from arco.learning_from_demo.probabilistic_encoding import ProbabilisticEncoding
from arco.learning_from_demo.trajectories import Trajectories


class ProbabilisticEncodingTest(unittest.TestCase):
    def test_probabilistic_encoding(self):

        base_path = pathlib.Path(__file__).parent.absolute()
        data_path = str(pathlib.Path(base_path, "data"))
        trajectories = Trajectories.from_dataset_file(data_path)
        pe = ProbabilisticEncoding(
            trajectories,
            max_nb_components=10,
            min_nb_components=2,
            iterations=1,
            random_state=0
        )
        # check best number GMM components
        self.assertEqual(pe.gmm.n_components, 3)
        # check norm of covariance matrices
        for i, norm in enumerate([249, 341, 308]):
            self.assertEqual(int(np.linalg.norm(pe.gmm.covariances_[i])), norm)


if __name__ == "__main__":
    unittest.main(exit=False)
