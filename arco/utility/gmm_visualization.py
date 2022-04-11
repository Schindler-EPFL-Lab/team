from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from arco.learning_from_demo.probabilistic_encoding import ProbabilisticEncoding


def plot_gmm(
    gmm: ProbabilisticEncoding,
    x_query: Optional[np.ndarray] = None,
    prediction: Optional[np.ndarray] = None
):
    """
    Plots obtained with GMM fitting

    :param gmm: the GMM fitted over the data
    :param x_query: query points vector
    :param prediction: regression line prediction
    """
    x = gmm.trajectories
    for i in range(1, np.shape(x)[1]):
        plt.figure(figsize=(10, 8))
        plt.scatter(x[:, 0], x[:, i], s=1, cmap="viridis", zorder=1, label="datapoints")
        plt.scatter(
            gmm.gmm.means_[:, 0],
            gmm.gmm.means_[:, i],
            c="black",
            s=200,
            alpha=0.5,
            zorder=2,
            label="Gaussian means",
        )
        plt.xlabel("time [s]", fontsize=16)
        plt.ylabel("joint angle [deg]", fontsize=16)
        plt.title(f"Joint {i} evolution", fontsize=20)

        w_factor = 0.2 / gmm.gmm.weights_.max()
        for pos, covar, w in zip(
            gmm.gmm.means_, gmm.gmm.covariances_, gmm.gmm.weights_
        ):
            covar = covar[0:i + 1:i, 0:i + 1:i]
            pos = pos[0:i + 1:i]
            draw_ellipse(pos, covar, alpha=w * w_factor)
        if x_query is not None and prediction is not None:
            plt.plot(
                x_query,
                prediction[:gmm.length_demo, i],
                color="r",
                linewidth=5,
                label="regression line",
            )
        plt.legend()
        plt.show()


def plot_js_distance(gmm_js: ProbabilisticEncoding):
    """
    Plots the mean and the std of the JS distance over the range of GMM components

    :param gmm_js: the GMM fittings over the data in the range of GMM components
    """
    min_idx = np.argmin(gmm_js.results)
    plt.errorbar(
        gmm_js.n_components_range,
        gmm_js.results,
        yerr=gmm_js.results_std,
        label="data mean and std"
    )
    plt.plot(
        gmm_js.nb_comp_js,
        gmm_js.results[min_idx],
        "o",
        c="r",
        markersize=10,
        label="optimal nb_components"
    )
    plt.legend()
    plt.title("Distance between Train and Test GMMs", fontsize=20)
    plt.xticks(gmm_js.n_components_range)
    plt.xlabel("Number of components", fontsize=16)
    plt.ylabel("Distance", fontsize=16)
    plt.show()


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """
    Draw an ellipse with a given position and covariance

    :param position: position of the ellipse center
    :param covariance: covariance of the ellipse
    :param ax: axes object to add ellipse to
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
    for std_dev in range(1, 4):
        ax.add_patch(
            Ellipse(position, std_dev * width, std_dev * height, angle, **kwargs)
        )
