import pathlib
import unittest

import numpy as np

from arco.learning_from_demo.data_preprocessing import DataPreprocessing


class PreprocessTest(unittest.TestCase):

    def test_preprocessing_steps(self):

        sampling_rate = 100
        base_path = pathlib.Path(__file__).parent.absolute()
        data_path = str(pathlib.Path(base_path, "data"))
        dp = DataPreprocessing(data_path, sampling_rate=sampling_rate)
        self.assertEqual(np.shape(dp.trajectories_to_align[0]), (519, 18))
        # test reference trajectory has been selected
        self.assertFalse(dp.reference.empty)
        # test number of trajectories to align is n - 1
        self.assertEqual(len(dp.trajectories_to_align), 1)
        dp.preprocessing()
        for data in dp.aligned_and_padded_trajectories:
            # test no duplicate rows remaining
            self.assertEqual(data[data.duplicated()].sum().any(), 0)
            # test sampling rate
            self.assertEqual(len(data) - 1, data.iloc[-1, 0] * sampling_rate)
            # test padding
            self.assertEqual(np.shape(dp.aligned_and_padded_trajectories)[1], 1027)


if __name__ == "__main__":
    unittest.main(exit=False)
