from typing import Optional

import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from team.aligned_trajectories import AlignedTrajectories
from team.js_component import JSComponent


class ProbabilisticEncoding:
    """
    Probabilistic encoding of the data through Gaussian mixture model probability
    distribution. This class allows to estimate the parameters of the best Gaussian
    mixture distribution according to the Jensen-Shannon (JS) metric

    ```python
    # probabilistic encoding of data with GMM number of components search space between
    2 and 10
    pe = ProbabilisticEncoding(
        trajectories, iterations=30, max_nb_components=10, min_nb_components=2
    )
    ```
    """

    def __init__(
        self,
        trajectories: AlignedTrajectories,
        iterations: int,
        min_nb_components: int = 2,
        max_nb_components: int = 10,
        random_state: Optional[int] = None,
        shuffle: bool = True,
    ) -> None:
        """
        :param trajectories: dataset with robot joint angle trajectories
            data has shape (nb_samples x demo_length x nb_joints + 1)
            the first column of each sample denotes the timestamp
        :param iterations: runs to compute statistics over JS metric
        :param min_nb_components: minimum number of GMM components
        :param max_nb_components: maximum number of GMM components
        :param random_state: reproducibility of GMM initialization for testing purposes
        :param shuffle: shuffle the train and test set while determining the best GMM
            Set to False to have reproducible runs.
        """
        self._iterations = iterations
        _, self.length_demo, self._nb_features = np.shape(
            trajectories.aligned_trajectories
        )
        # dataset as a NumPy array of shape (n_samples_per_trajectories *
        # n_trajectories, n_features)
        self.trajectories = trajectories.aligned_trajectories.reshape(
            -1, self._nb_features
        )
        # js distances dictionary
        self.js_metric_results: dict[int, JSComponent] = {}
        # chosen number of components
        self.nb_comp_js = 0
        # number of components selection
        self.gmm = self._select_gmm_js_distance(
            max_nb_components=max_nb_components,
            min_nb_components=min_nb_components,
            random_state=random_state,
            shuffle=shuffle,
        )

    def _gmm_fitting(
        self,
        nb_components: int,
        cov_type: str = "full",
        init_type: str = "kmeans",
        random_state: Optional[int] = None,
    ) -> GaussianMixture:
        """
        Fits a Gaussian Mixture Model on the data

        :param nb_components: number of GMM components
        :param cov_type: covariance type (e.g spherical, full, ..)
        :param init_type: parameters initialization type (e.g kmeans, random)
        :param random_state: reproducibility of GMM initialization for testing purposes
        :return: the fitted GMM mixture on the data
        """
        gmm = GaussianMixture(
            n_components=nb_components,
            covariance_type=cov_type,
            init_params=init_type,
            random_state=random_state,
        )
        # the fitted mixture
        return gmm.fit(self.trajectories)

    def _select_gmm_js_distance(
        self,
        max_nb_components: int,
        min_nb_components: int,
        random_state: Optional[int] = None,
        shuffle: bool = True,
    ) -> GaussianMixture:
        """
        Computes the Jensen-Shannon (JS) metric. The lesser is the JS-distance between
        the two GMMs, the more the GMMs agree on how to fit the data.
        Returns the best GMM fitting with the identified best number of GMM components

        :param min_nb_components: min components number to define range of search space.
        :param max_nb_components: max components number to define range of search space.
        :param random_state: reproducibility of GMM initialization for testing purposes.
        :param shuffle: shuffle the train and test set while determining the best GMM.
        :return: the best fitted GMM mixture on the data according to JS distance score.
        """
        # check valid range for number components
        if (
            max_nb_components <= min_nb_components
            or max_nb_components < 2
            or min_nb_components < 2
        ):
            raise RuntimeError(
                "Invalid range for search space! max_nb_components must "
                "be larger or equal to min_nb_components and "
                "min_nb_components not smaller than 2"
            )
        # search space range
        n_components_range = range(min_nb_components, max_nb_components)
        # clear js_metric_results
        self.js_metric_results = {}
        # loop over range
        for n in n_components_range:
            component = JSComponent(n)
            # loop over number runs
            for _ in range(self._iterations):
                train, test = train_test_split(
                    self.trajectories,
                    test_size=0.5,
                    random_state=random_state,
                    shuffle=shuffle,
                )
                # fit over the train and test datasets
                gmm_train = GaussianMixture(n, random_state=random_state).fit(train)
                gmm_test = GaussianMixture(n, random_state=random_state).fit(test)
                # compute the JS distance between the two datasets
                component.append(self._js_metric(gmm_train, gmm_test))

            self.js_metric_results[n] = component

        # identify statistically significant best GMM nb_components
        self.nb_comp_js = self._statistically_significant_component()
        return self._gmm_fitting(
            nb_components=self.nb_comp_js, random_state=random_state
        )

    def compute_scores_bic_criterion(
        self,
        max_nb_components: int,
        min_nb_components: int,
    ) -> dict[str, list]:
        """
        Computes the BIC criterion for the range of GMM components.

        :param min_nb_components: min components number to define range of search space
        :param max_nb_components: max components number to define range of search space
        :return: the BIC scores
        """
        # check valid range for number components
        if (
            max_nb_components <= min_nb_components
            or max_nb_components < 2
            or min_nb_components < 2
        ):
            raise RuntimeError(
                "Invalid range for search space! max_nb_components must "
                "be larger or equal to min_nb_components and "
                "min_nb_components not smaller than 2"
            )
        # search space range
        n_components_range = range(min_nb_components, max_nb_components)
        # clear bic_scores
        bic_scores = {}
        # loop over range
        mean_components = []
        std_components = []
        for n in n_components_range:
            scores_component = []
            # loop over number runs
            for _ in range(self._iterations):
                # fit GMM
                gmm_fit = GaussianMixture(n).fit(self.trajectories)
                # compute the JS distance between the two datasets
                scores_component.append(gmm_fit.bic(self.trajectories))

            mean_components.append(np.mean(scores_component))
            std_components.append(np.std(scores_component))
        bic_scores["means"] = mean_components
        bic_scores["stds"] = std_components

        return bic_scores

    def _statistically_significant_component(self) -> int:
        """
        Compares the JS metric samples [self.js_metric_results] to statistically infer
        the best number of components to pick.

        Alternative hypothesis: the min JS mean distance is smaller than all the others
                                JS mean distances
        Null hypothesis: the min JS mean distance is not smaller than all the others JS
                         mean distances

        We use the Welch-test to perform the hypothesis test. alpha is 0.05,
        corresponding to a 5% chance the results occurred at random. If the observed
        p-value is less than alpha, then the null hypothesis is rejected and the
        observations are statistically significant.

        :return: the number of GMM components to consider
        """

        k_star_component: Optional[JSComponent] = None

        for _, component in self.js_metric_results.items():
            if k_star_component is None or component.mean < k_star_component.mean:
                k_star_component = component

        assert k_star_component is not None
        for k_component, component in self.js_metric_results.items():
            if k_component == k_star_component.component_nb:
                continue
            # compute p-value
            # Explanation of the formula for Welch test with distributions.
            # http://homework.uoregon.edu/pub/class/es202/ztest.html#:~:text=The%20simplest%20way%20to%20compare,is%20via%20the%20Z%2Dtest.&text=The%20error%20in%20the%20mean,mean%20value%20for%20that%20population.
            # Test if k is strictily greater than k*. The alternate hypothesis is k* < k
            # noqa

            _, p_value_greater = stats.ttest_ind(
                component.values,
                k_star_component.values,
                equal_var=False,
                alternative="greater",
            )
            if p_value_greater < 0.05:
                # k is strictily greater than k* and we stop the process
                continue

            # Test if k* is strictily greater than k. The alternate hypothesis is k < k*
            _, p_value_lesser = stats.ttest_ind(
                k_star_component.values,
                component.values,
                equal_var=False,
                alternative="greater",
            )

            if p_value_lesser < 0.05 or component.std < k_star_component.std:
                # Etiher k* is strictily greater than k and k is more optiomal than k*
                # or we cannot conclude on which component is best and we keep the one
                # with the smallest standard deviation.
                k_star_component = component

        return k_star_component.component_nb

    def count_collapsed_gmm(self, gmm: GaussianMixture) -> int:
        """
        Counts how many GMM components have collapsed on single or few datapoints
        during EM. The reference covariance [ref_cov] is built with a diagonal matrix
        whose entries are 1% of the motion magnitude along the respective axis.

        :param gmm: GMM fit of the data
        :return: the number of degenerated GMM components
        """

        d_time = self.length_demo / 100
        d_joints = (
            np.max(self.trajectories, axis=0) - np.min(self.trajectories, axis=0)
        ) / 100
        ref_cov = np.array([[d_joints, 0], [0, d_time]])
        epsilon = np.linalg.norm(ref_cov)
        nb_collapsed = 0
        for gc in gmm.covariances_:
            if np.linalg.norm(gc) < epsilon:
                nb_collapsed += 1
        return nb_collapsed

    @staticmethod
    def _js_metric(
        gmm_p: GaussianMixture, gmm_q: GaussianMixture, n_samples: int = 10**5
    ) -> float:
        """
        Calculates the Jensen-Shannon divergence metric

        :param gmm_p: GMM fitted over training set
        :param gmm_q: GMM fitted over testing set
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
        numerator = round(
            (
                log_p_x.mean()
                - (log_mix_x.mean() - np.log(2))
                + log_q_y.mean()
                - (log_mix_y.mean() - np.log(2))
            ),
            6,
        )
        if numerator < 0:
            raise RuntimeError("numerator of JS divergence is less than 0")
        js_distance = np.sqrt(numerator / 2)

        return js_distance
