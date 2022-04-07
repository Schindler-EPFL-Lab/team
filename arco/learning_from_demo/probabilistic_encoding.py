import itertools

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


class ProbabilisticEncoding:
    gmm: GaussianMixture
    best_gmm: GaussianMixture

    def __init__(self, data: np.ndarray, max_k: int) -> None:
        """

        :param data:
        :param max_k:
        """

        self._data = data
        self.nb_demo, self.length_demo, self.nb_features = np.shape(data)
        self.cv_types = ["spherical", "tied", "diag", "full"]
        self.cov_type, self.gmm_components = self.select_gmm_bic_criterion(
            max_nb_components=max_k, cv_types=self.cv_types, to_plot=True
        )

    def gmm_fitting(self, nb_components: int, cov_type: str, init_type: str = "kmeans"):
        """
        Fits a Gaussian Mixture Model with given hyper-parameters on the data

        :param nb_components: number of GMM components
        :param cov_type: covariance type (e.g spherical, full, ..)
        :param init_type: parameters initialization type (e.g kmeans, random)
        :return: the fitted mixture on the data
        """
        # dataset as a NumPy array of shape (n_samples, n_features)
        x = self._data.reshape(-1, self.nb_features)
        gmm = GaussianMixture(n_components=nb_components, covariance_type=cov_type,
                              init_params=init_type)
        # the fitted mixture
        self.gmm = gmm.fit(x)

    def select_gmm_bic_criterion(self, max_nb_components: int, cv_types: list[str],
                                 to_plot: bool = False,) -> (str, int):

        lowest_bic = np.infty
        bic = []
        n_components_range = range(1, max_nb_components)
        # dataset as a NumPy array of shape (n_samples, n_features)
        x = self._data.reshape(-1, self.nb_features)
        # loop over all of covariance types
        for cv_type in cv_types:
            # loop over range of GMM components
            for n_components in n_components_range:
                # fit a Gaussian mixture with EM
                self.gmm_fitting(n_components, cv_type)
                bic.append(self.gmm.bic(x))
                if bic[-1] < 0:
                    bic[-1] = np.abs(bic[-1]) + lowest_bic
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    self.best_gmm = self.gmm

        # Plot the BIC scores
        if to_plot:
            self.plot_bic_score(bic, cv_types, n_components_range)

        return self.best_gmm.covariance_type, self.best_gmm.n_components

    @staticmethod
    def plot_bic_score(bic: list, cv_types: list[str], n_components_range: range):
        bic = np.array(bic)
        color_iter = itertools.cycle(
            ["navy", "turquoise", "cornflowerblue", "darkorange"])
        bars = []
        plt.figure(figsize=(8, 6))
        spl = plt.subplot(2, 1, 1)
        for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
            x_pos = np.array(n_components_range) + 0.2 * (i - 2)
            bars.append(
                plt.bar(
                    x_pos,
                    bic[i * len(n_components_range): (i + 1) * len(n_components_range)],
                    width=0.2,
                    color=color,
                )
            )
        plt.xticks(n_components_range)
        plt.ylim([np.min(bic) * 1.01 - 0.01 * np.max(bic), np.max(bic)])
        plt.title("BIC score per model")
        x_pos = (
                np.mod(bic.argmin(), len(n_components_range))
                + 0.65
                + 0.2 * np.floor(bic.argmin() / len(n_components_range))
        )
        plt.text(x_pos, np.min(bic) * 0.97 + 0.03 * np.max(bic), "*", fontsize=14)
        spl.set_xlabel("Number of components")
        spl.legend([b[0] for b in bars], cv_types)
