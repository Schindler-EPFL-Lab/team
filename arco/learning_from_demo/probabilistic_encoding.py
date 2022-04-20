from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from matplotlib.patches import Ellipse


class ProbabilisticEncoding:
    """
    Probabilistic encoding of the data through Gaussian mixture model probability
    distribution. This class allows to estimate the parameters of the best Gaussian
    mixture distribution according to the Jensen-Shannon (JS) metric

    ```python
    # probabilistic encoding of data with GMM number of components search space between
    2 and 10, required plot and final fitting plot
    pe = ProbabilisticEncoding(
        d, iterations=30, max_nb_components=10, min_nb_components=2, to_plot=True
    )
    pe.plot_gmm()
    ```

    :param data: dataset to fit
    :param iterations: runs to compute statistics over JS metric
    :param min_nb_components: minimum number of GMM components allowed to model the data
    :param max_nb_components: maximum number of GMM components allowed to model the data
    :param to_plot: boolean to assess if the plot is required or not
    :param random_state: reproducibility of GMM initialization for testing purposes
    """

    def __init__(
        self,
        data: np.ndarray,
        iterations: int,
        min_nb_components: int = 2,
        max_nb_components: int = 10,
        to_plot: bool = False,
        random_state: Optional[int] = None
    ) -> None:
        self._data = data
        self._iterations = iterations
        _, _, self._nb_features = np.shape(self._data)
        self.gmm = self._select_gmm_js_distance(
            max_nb_components=max_nb_components,
            min_nb_components=min_nb_components,
            to_plot=to_plot,
            random_state=random_state,
        )

    def _gmm_fitting(
        self,
        nb_components: int,
        cov_type: str = "full",
        init_type: str = "kmeans",
        random_state: Optional[int] = None
    ) -> GaussianMixture:
        """
        Fits a Gaussian Mixture Model with given hyper-parameters on the data

        :param nb_components: number of GMM components
        :param cov_type: covariance type (e.g spherical, full, ..)
        :param init_type: parameters initialization type (e.g kmeans, random)
        :param random_state: reproducibility of GMM initialization for testing purposes
        :return: the fitted GMM mixture on the data
        """
        # dataset as a NumPy array of shape (n_samples, n_features)
        x = self._data.reshape(-1, self._nb_features)
        gmm = GaussianMixture(
            n_components=nb_components,
            covariance_type=cov_type,
            init_params=init_type,
            random_state=random_state,
        )
        # the fitted mixture
        return gmm.fit(x)

    def _select_gmm_js_distance(
        self,
        max_nb_components: int,
        min_nb_components: int = 2,
        to_plot: bool = False,
        random_state: Optional[int] = None
    ) -> GaussianMixture:
        """
        Computes the Jensen-Shannon (JS) metric. The lesser is the JS-distance between
        the two GMMs, the more the GMMs agree on how to fit the data.
        Returns the best GMM fitting with the identified best number of GMM components

        :param min_nb_components: min components number to define range of search space.
               Defaults to 2 if less than 2.
        :param max_nb_components: max components number to define range of search space.
               Defaults to 2 if less than 2.
        :param to_plot: boolean assessing if the BIC scores plot is required or not
        :param random_state: reproducibility of GMM initialization for testing purposes
        :return: the best fitted GMM mixture on the data according to JS distance score
        """
        # check valid range for number components
        if min_nb_components <= 2:
            min_nb_components = 2
        if max_nb_components <= 2:
            max_nb_components = 2
        if max_nb_components <= min_nb_components:
            raise RuntimeError("Invalid range for search space!")
        # search space range
        n_components_range = range(min_nb_components, max_nb_components)
        # dataset as a NumPy array of shape (n_samples, n_features)
        x = self._data.reshape(-1, self._nb_features)
        # runs for standard deviation
        results = []
        res_sigs = []
        # loop over range
        for n in n_components_range:
            dist = []
            # loop over number runs
            for iteration in range(self._iterations):
                train, test = train_test_split(
                    x, test_size=0.5, random_state=random_state
                )
                # fit over the train and test datasets
                gmm_train = GaussianMixture(n, random_state=random_state).fit(train)
                gmm_test = GaussianMixture(n, random_state=random_state).fit(test)
                # compute the JS distance between the two datasets
                dist.append(self._js_metric(gmm_train, gmm_test))

            results.append(np.mean(dist))
            res_sigs.append(np.std(dist))

        # identify minimum in the JS distance corresponding to best nb_components
        min_idx = np.argmin(results)
        nb_comp_js = n_components_range[min_idx]
        if to_plot:
            self._plot_js_distance(n_components_range, results, res_sigs, nb_comp_js)

        return self._gmm_fitting(nb_components=nb_comp_js, random_state=random_state)

    def plot_gmm(self):
        """
        Plots obtained with GMM fitting
        """
        x = self._data.reshape(-1, self._nb_features)
        for i in range(1, np.shape(x)[1]):
            plt.figure(figsize=(10, 8))
            plt.scatter(x[:, 0], x[:, i], s=1, cmap="viridis", zorder=1)
            plt.scatter(
                self.gmm.means_[:, 0],
                self.gmm.means_[:, i],
                c="black",
                s=300,
                alpha=0.5,
                zorder=2,
            )
            plt.xlabel("time [s]", fontsize=16)
            plt.ylabel("joint angle [deg]", fontsize=16)
            plt.title(f"Joint {i} evolution", fontsize=20)

            w_factor = 0.2 / self.gmm.weights_.max()
            for pos, covar, w in zip(
                self.gmm.means_, self.gmm.covariances_, self.gmm.weights_
            ):
                covar = covar[0: i + 1: i, 0: i + 1: i]
                pos = pos[0: i + 1: i]
                self._draw_ellipse(pos, covar, alpha=w * w_factor)
            plt.legend(["datapoints", "Gaussian means", "Gaussian covariances"])
            plt.show()

    @staticmethod
    def _js_metric(
        gmm_p: GaussianMixture, gmm_q: GaussianMixture, n_samples: int = 10 ** 5
    ) -> float:
        """
        Calculates the Jensen-Shannon divergence metric

        :param gmm_p:
        :param gmm_q:
        :param n_samples: number of samples extracted from the distribution
        :return: the JS metric of the configuration
        """
        # sampled datapoints from distribution
        x = gmm_p.sample(n_samples)[0]
        # weighted log probabilities for each sample with fitting gmm_p
        log_p_x = gmm_p.score_samples(x)
        # weighted log probabilities for each sample with fitting gmm_q
        log_q_x = gmm_q.score_samples(x)
        # mixed weighted log probabilities
        log_mix_x = np.logaddexp(log_p_x, log_q_x)

        y = gmm_q.sample(n_samples)[0]
        log_p_y = gmm_p.score_samples(y)
        log_q_y = gmm_q.score_samples(y)
        log_mix_y = np.logaddexp(log_p_y, log_q_y)
        # js divergence metric
        js_distance = np.sqrt(
            (
                log_p_x.mean()
                - (log_mix_x.mean() - np.log(2))
                + log_q_y.mean()
                - (log_mix_y.mean() - np.log(2))
            )
            / 2
        )

        return js_distance

    @staticmethod
    def _plot_js_distance(
        n_components_range: range,
        results: list[np.ndarray],
        res_sigs: list[np.ndarray],
        nb_comp_js: int
    ):
        """
        Plots the mean and the std of the JS distance over the range of GMM components

        :param n_components_range: range of GMM components to consider
        :param results: list of JS distance means
        :param res_sigs: list of JS distance std
        :param nb_comp_js: chosen number of components
        """
        min_idx = np.argmin(results)
        plt.plot(nb_comp_js, results[min_idx], "o", c="r", markersize=10)
        plt.errorbar(n_components_range, results, yerr=res_sigs)
        plt.legend(["data mean and std", "optimal nb_components"])
        plt.title("Distance between Train and Test GMMs", fontsize=20)
        plt.xticks(n_components_range)
        plt.xlabel("Number of components", fontsize=16)
        plt.ylabel("Distance", fontsize=16)
        plt.show()

    @staticmethod
    def _draw_ellipse(position, covariance, ax=None, **kwargs):
        """
        Draw an ellipse with a given position and covariance
        """
        ax = ax or plt.gca()
        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # Draw the Ellipse
        for nsig in range(1, 4):
            ax.add_patch(
                Ellipse(position, nsig * width, nsig * height, angle, **kwargs)
            )
