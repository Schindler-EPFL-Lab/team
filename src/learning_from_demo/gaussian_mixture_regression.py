import os
from pathlib import Path

import numpy as np
from gmr import GMM

from learning_from_demo.probabilistic_encoding import ProbabilisticEncoding
from learning_from_demo.aligned_trajectories import AlignedTrajectories


class GMR:
    """
    Calculates the Gaussian Mixture Regression line from the probability encoding of the
    data. Each time step is used as query point and the corresponding predictions denote
    the regression line

    ```python
    # probabilistic encoding
    pe = ProbabilisticEncoding(trajectories=trajectories, iterations=10,
         max_nb_components=10, min_nb_components=2)
    # extraction of regression line from probability distributions and plot shown
    gmr = GMR(trajectories=trajectories, prob_encoding=pe)
    regression = gmr.prediction
    ```

    :param trajectories: preprocessed dataset
    :param prob_encoding: probabilistic encoding of the data through GMM models
    """

    def __init__(
        self, trajectories: AlignedTrajectories, prob_encoding: ProbabilisticEncoding
    ) -> None:
        self._trajectories = trajectories
        self._gmm = GMM(
            n_components=prob_encoding.gmm.n_components,
            priors=prob_encoding.gmm.weights_,
            means=prob_encoding.gmm.means_,
            covariances=np.array(prob_encoding.gmm.covariances_),
        )
        self.prediction = self._predict_regression()

    def _predict_regression(self) -> np.ndarray:
        """
        Computes the Gaussian Mixture Regression line

        :return: the predicted regression line
        """
        # extract first feature -> time vector (query points)
        x1 = self._trajectories.timestamps.reshape(-1, 1)
        x1_index = [0]
        # make predictions for all features
        y_predicted_mean = self._gmm.predict(x1_index, x1)
        # stack time vector with predictions
        y_predicted_mean = np.hstack((x1, y_predicted_mean))
        return y_predicted_mean

    def save_regression_data(self, dir_path: Path) -> None:
        """
        Saves regression function data. If the destination folder doesn't exist it
        creates it.

        :param dir_path: path to the directory where the data will be saved
        """
        # create the parent folder if it does not exist
        if not os.path.exists(dir_path):
            try:
                dir_path.mkdir()
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Parent folder does not exists, please check the "
                    "consistency of the provided path!"
                )
        file_path = dir_path.joinpath("regression.npy")
        if os.path.exists(file_path):
            raise FileExistsError("Not allowed to override an existing file!")
        np.save(str(file_path), self.prediction)
