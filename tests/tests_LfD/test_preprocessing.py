import pathlib
import unittest

import numpy as np
import pandas as pd

from learning_from_demo.aligned_trajectories import AlignedTrajectories
from learning_from_demo.data_preprocessing import DataPreprocessing


class PreprocessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sampling_rate = 100
        base_path = pathlib.Path(__file__).parent.absolute()
        data_path = str(pathlib.Path(base_path, "data"))
        trajectories_list = AlignedTrajectories._load_data(data_path)
        cls.dp = DataPreprocessing(trajectories_list, sampling_rate=cls.sampling_rate)

    def test_reference_selection(self) -> None:
        self.assertEqual(self.dp.reference_index, 0)

    def test_read_file_preprocessing(self) -> None:
        traj_length = {(728, 10), (776, 10), (519, 10)}
        traj_read_len = set()
        for traj in self.dp.trajectories_to_align:
            traj_read_len.add(np.shape(traj.trajectory))
        self.assertEqual(traj_length, traj_read_len)

    def test_preprocessing_steps(self) -> None:
        self.dp.preprocessing()
        aligned_traj = AlignedTrajectories.from_list_trajectories(
            self.dp.aligned_and_padded_trajectories
        )
        self.assertEqual(np.shape(aligned_traj.aligned_trajectories)[0], 3)
        for trajectory in self.dp.aligned_and_padded_trajectories:
            data = pd.DataFrame(trajectory.trajectory)
            # test no duplicate rows remaining
            self.assertEqual(data[data.duplicated()].sum().any(), 0)
            # test sampling rate
            self.assertEqual(
                len(data) - 1, round(data.iloc[-1, 0] * self.sampling_rate)
            )
            # test padding
            first_traj = self.dp.aligned_and_padded_trajectories[0]
            self.assertEqual(np.shape(first_traj.trajectory)[0], 1285)
