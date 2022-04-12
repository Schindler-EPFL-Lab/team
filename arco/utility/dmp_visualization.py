import numpy as np
import matplotlib.pyplot as plt

from arco.learning_from_demo.dynamical_movement_primitives import (
    DynamicMovementPrimitives,
)


def plotting(dmp: DynamicMovementPrimitives) -> None:
    """
    Generates the plots of the dynamic movement primitives fitting

    :param dmp: the dynamic movement primitives model
    """

    len_demo, nb_joints = np.shape(dmp.T)[:-1]
    # plotting
    time = dmp.dt * np.arange(len_demo)
    for i in range(nb_joints):

        fig, axs = plt.subplots(4, 3, figsize=(15, 15))
        # plot position, velocity, acceleration vs.target
        axs[0, 0].plot(time, dmp.y[:, i, 0])
        axs[0, 0].plot(time, dmp.T[:, i, 0])
        axs[0, 0].set_title("y")
        axs[0, 1].plot(time, dmp.y[:, i, 1])
        axs[0, 1].plot(time, dmp.T[:, i, 1])
        axs[0, 1].set_title("yd")
        axs[0, 2].plot(time, dmp.y[:, i, 2])
        axs[0, 2].plot(time, dmp.T[:, i, 2])
        axs[0, 2].set_title("ydd")

        # plot internal states
        axs[1, 0].plot(time, dmp.z_history[:, i, 0])
        axs[1, 0].set_title("z")
        axs[1, 1].plot(time, dmp.z_history[:, i, 1])
        axs[1, 1].set_title("zd")
        axs[1, 2].plot(time, dmp.psi_history[:, i, :])
        axs[1, 2].set_title("Weighting Kernels")
        axs[2, 0].plot(time, dmp.v_history[:, i, 0])
        axs[2, 0].set_title("v")
        axs[2, 1].plot(time, dmp.v_history[:, i, 1])
        axs[2, 1].set_title("vd")
        axs[2, 2].plot(time, dmp.w_history[:, i])
        axs[2, 2].set_title("Linear Model Weights over Time")
        axs[3, 0].plot(time, dmp.x_history[:, i, 0])
        axs[3, 0].set_title("x")
        axs[3, 1].plot(time, dmp.x_history[:, i, 1])
        axs[3, 1].set_title("xd")
        axs[3, 2].plot(dmp.w_history[-1, i, :])
        axs[3, 2].set_title("Weights")
        # set the spacing between subplots
        plt.subplots_adjust(
            left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4
        )
    plt.show()
