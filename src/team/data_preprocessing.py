import numpy as np
import pandas as pd
from dtaidistance import dtw, dtw_ndim

from team.trajectory import Trajectory


class DataPreprocessing:
    """
    Preprocesses the dataset to align demonstrations in time, to up-sample at desired
    sampling frequency and to have the same time-series length

    ```python
    dp = DataPreprocessing(traj_to_align, sampling_rate)
    # preprocessing
    dp.preprocessing()
    ```

    :param traj_to_align: trajectories to align
    :param sampling_rate: the desired sampling frequency
    """

    def __init__(self, traj_to_align: list[Trajectory], sampling_rate: int) -> None:
        self.trajectories_to_align = traj_to_align
        # minimal cumulative distance demonstration as reference
        self.reference_index = self.select_reference_index()
        self.reference = self.trajectories_to_align[self.reference_index]
        self._sampling_rate = sampling_rate
        # final output of the algorithm
        self.aligned_and_padded_trajectories = [self.reference]

    @staticmethod
    def _extend_duplicates(traj: Trajectory, av_sampling: float) -> None:
        """
        Extends the timestamps of the duplicated values (i.e generates new datapoints
        from duplicated values with the same information but different timestamps)

        :param traj: the trajectory to extend
        :param av_sampling: average sampling rate of the data
        """

        # No copy of the numpy array is done when converting to a DataFrame, and thus,
        # we are still working on the trajectory.
        df = pd.DataFrame(traj.trajectory)
        for i, is_duplicated in enumerate(df.duplicated()):
            if is_duplicated:
                df.iloc[i:, 0] = df.iloc[i:, 0] + (1 / av_sampling)

    @staticmethod
    def _pad_to_same_length(trajectories: list[Trajectory]) -> None:
        """
        Detects the maximum trajectory length and then pads all the other demonstrations
        with their respective end value to have the same length

        :param trajectories: list of Trajectory objects
        :return: list with padded dataframes
        """
        max_len = len(max(trajectories, key=len))
        for trajectory in trajectories:
            trajectory.pad_end(max_len)

    def select_reference_index(self) -> int:
        """
        Computes the distance matrix for each pair of demonstrations in the dataset.
        Selects as reference the demonstration with the minimal cumulative distance
        (i.e the most centered with respect to the dataset).

        :return: the index of the reference demonstration in
            [self.trajectories_to_align]
        """
        # put data in the distance_matrix_fast method required format
        time_series = [traj.tcp for traj in self.trajectories_to_align]
        ds = dtw.distance_matrix_fast(time_series)
        # sum over one axis (ds matrix is symmetric)
        cumulative_dist = np.sum(ds, axis=1)
        return np.argwhere(cumulative_dist == np.min(cumulative_dist))[0][0]

    def _align_data(self) -> None:
        """
        Aligns demonstrations with respect to the reference one using the dynamic
        time warping algorithm (DTW)

        """
        # end effector position information considered
        tcp_ref = self.reference.tcp
        path = None
        for i, trajectory in enumerate(self.trajectories_to_align):
            if i == self.reference_index:
                continue
            path = dtw_ndim.warping_path(
                from_s=trajectory.tcp, to_s=tcp_ref, window=10, psi=2
            )
            aligned_path = [p[0] for p in path]
            # found transformation applied to original dataframe
            aligned_trajectory = Trajectory(trajectory.trajectory[aligned_path])
            self.aligned_and_padded_trajectories.append(aligned_trajectory)

    def _stretch_duplicates(self) -> None:
        """
        Extends in time the rows of the dataframe that have been duplicated so that
        demonstrations can align in time

        """
        for traj in self.aligned_and_padded_trajectories:
            # extends dataframe with time increments
            DataPreprocessing._extend_duplicates(traj, traj.average_sampling)

    def _upsampling_padding(self) -> None:
        """
        Upsamples the obtained trajectories at the desired sampling frequency and pads
        the end of each demonstration to have the same time-series length

        """
        for data in self.aligned_and_padded_trajectories:
            data.upsample(self._sampling_rate)
        self._pad_to_same_length(self.aligned_and_padded_trajectories)

    def preprocessing(self) -> None:
        """
        Runs all the preprocessing steps
        """
        self._align_data()
        self._stretch_duplicates()
        self._upsampling_padding()
