import pandas as pd
import numpy as np
from dtaidistance import dtw, dtw_ndim
from scipy import interpolate


class DataPreprocessing:
    """
    Preprocesses the dataset to align demonstrations in time, to up-sample at desired
    sampling frequency and to have the same time-series length

    ```python
    dp = DataPreprocessing(data_path, sampling_rate)
    # preprocessing
    dp.preprocessing()
    ```

    :param traj_to_align: trajectories to align
    :param sampling_rate: the desired sampling frequency
    """

    def __init__(
        self, traj_to_align: list[pd.DataFrame], sampling_rate: int
    ) -> None:
        self.trajectories_to_align = traj_to_align
        # minimal cumulative distance demonstration as reference
        self._df_ref = self._select_reference_demo()
        self._sampling_rate = sampling_rate
        self._aligned_data_list = []
        self._extended_dfs = []
        self._upsampled_data = []
        # final output of the algorithm
        self.aligned_and_padded_trajectories = []

    @staticmethod
    def _extend_duplicates(df: pd.DataFrame, av_sampling: float) -> pd.DataFrame:
        """
        Extends the timestamps of the duplicated values (i.e generates new datapoints
        from duplicated values with the same information but different timestamps)

        :param df: the dataframe to extend
        :param av_sampling: average sampling rate of the data
        :return: the extended dataframe
        """
        for i, is_duplicated in enumerate(df.duplicated()):
            if is_duplicated:
                df.iloc[i:, 0] = df.iloc[i:, 0] + (1 / av_sampling)
        return df

    @staticmethod
    def _upsample_data(df: pd.DataFrame, des_freq: int) -> pd.DataFrame:
        """
        Computes a linear interpolation function fitting the joints to the timestamp and
        then generates the missing values applying the interpolating function to the
        time vector sampled at des_freq

        :param df: the dataframe to up-sample
        :param des_freq: the desired sampling frequency
        :return: the up-sampled dataframe
        """
        df.iloc[:, 0] = df.iloc[:, 0] - df.iloc[0, 0]
        timestamp = df["timestamp"].to_numpy()
        joints = df[
            ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        ].to_numpy()
        f = interpolate.interp1d(timestamp, joints, axis=0)
        time_new = np.arange(0, int(timestamp[-1] * des_freq) / des_freq, 1 / des_freq)
        # use interpolation function returned by `interp1d` to generate new data points
        joints_new = f(time_new)
        data = np.append(time_new.reshape(-1, 1), joints_new, axis=1)
        return pd.DataFrame(
            data,
            columns=[
                "timestamp",
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
            ],
        )

    @staticmethod
    def _pad_to_same_length(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """
        Detects the maximum trajectory length and then pads all the other demonstrations
        with their respective end value to have the same length

        :param dfs: list of dataframes
        :return: list with padded dataframes
        """
        padded_data = []
        max_len = len(max(dfs, key=len))
        for df in dfs:
            if len(df) != max_len:
                # computes number of missing data points
                nb_samples = max_len - len(df)
                # duplicates nb_samples time the last row
                df = pd.concat([df, pd.concat([df.iloc[-1:]] * nb_samples)])
                # updates the timestamps
                df.iloc[-nb_samples:, 0] = df.iloc[-nb_samples:, 0] + np.linspace(
                    0.01, 0.01 * nb_samples, nb_samples
                )
                df.reset_index(drop=True, inplace=True)
            padded_data.append(df)
        return padded_data

    @property
    def reference(self):
        return self._df_ref

    def _select_reference_demo(self) -> pd.DataFrame:
        """
        Computes the distance matrix for each pair of demonstrations in the dataset.
        Selects as reference the demonstration with the minimal cumulative distance
        (i.e the most centered with respect to the dataset) and it removes it from the
        demonstration dataset to avoid self comparison in the following step.

        :return: the selected demonstration
        """
        # put data in the distance_matrix_fast method required format
        time_series = [
            df[["tcp_x", "tcp_y", "tcp_z"]].to_numpy()
            for df in self.trajectories_to_align
        ]
        ds = dtw.distance_matrix_fast(time_series)
        # sum over one axis (ds matrix is symmetric)
        cumulative_dist = np.sum(ds, axis=1)
        idx_min = np.argwhere(cumulative_dist == np.min(cumulative_dist))[0][0]
        # remove and return selected reference
        return self.trajectories_to_align.pop(idx_min)

    def _align_data(self) -> None:
        """
        Aligns demonstrations with respect to the reference one using the dynamic
        time warping algorithm (DTW)

        """
        self._aligned_data_list = []
        # end effector position information considered
        tcp_ref = np.array(self._df_ref[["tcp_x", "tcp_y", "tcp_z"]])
        path = None
        for df_2 in self.trajectories_to_align:
            tcp2 = np.array(df_2[["tcp_x", "tcp_y", "tcp_z"]])
            d, paths = dtw_ndim.warping_paths(tcp_ref, tcp2)
            # best matching transformation
            path = dtw.best_path(paths)
            path_df_2 = [p[1] for p in path]
            # found transformation applied to original dataframe
            df_2 = df_2.iloc[path_df_2]
            self._aligned_data_list.append(df_2)
        # reference demonstration added
        path_df_ref = [p[0] for p in path]
        self._df_ref = self._df_ref.iloc[path_df_ref]
        self._aligned_data_list.append(self._df_ref)

    def _stretch_duplicates(self) -> None:
        """
        Extends in time the rows of the dataframe that have been duplicated so that
        demonstrations can align in time

        """
        self._extended_dfs = []
        for align_df in self._aligned_data_list:
            # compute the average sampling frequency of the acquired demo
            average_sampling = (
                len(align_df) - align_df.duplicated().sum()
            ) / align_df.iloc[-1, 0]
            # extends dataframe with time increments
            df_ext = self._extend_duplicates(align_df, average_sampling)
            self._extended_dfs.append(df_ext)

    def _upsampling_padding(self) -> None:
        """
        Upsamples the obtained trajectories at the desired sampling frequency and pads
        the end of each demonstration to have the same time-series length

        """
        self._upsampled_data = []
        for data in self._extended_dfs:
            ups_data = self._upsample_data(data, self._sampling_rate)
            self._upsampled_data.append(ups_data)
        self.aligned_and_padded_trajectories = self._pad_to_same_length(
            self._upsampled_data
        )

    def preprocessing(self) -> None:
        """
        Runs all the preprocessing steps
        """
        self._align_data()
        self._stretch_duplicates()
        self._upsampling_padding()
