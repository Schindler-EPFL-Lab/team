import numpy as np
import matplotlib.pyplot as plt

from learning_from_demo.dynamical_movement_primitives import DynamicMovementPrimitives


def plotting(dmp: DynamicMovementPrimitives) -> None:
    """
    Generates the plots of the dynamic movement primitives fitting

    :param dmp: the dynamic movement primitives model
    """

    len_demo, nb_joints = np.shape(dmp.T)[:-1]
    # plotting
    time = dmp.dt * np.arange(len_demo)
    for i in range(nb_joints):

        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        # plot position, velocity, acceleration vs.target
        axs[0, 0].plot(time, dmp.y[:, i, 0], label="output")
        axs[0, 0].plot(time, dmp.T[:, i, 0], label="regression")
        axs[0, 0].plot(time[-1], dmp._G[i], color='green', marker='o', markersize=6)
        axs[0, 0].set_title("y", fontsize=20)
        axs[0, 0].set_ylabel("Joint angles [$deg$]", fontsize=14)
        axs[0, 0].set_xlabel("Time [s]", fontsize=14)
        axs[0, 0].legend(fontsize=12)
        axs[0, 1].plot(time, dmp.y[:, i, 1], label="output")
        axs[0, 1].plot(time, dmp.T[:, i, 1], label="regression")
        axs[0, 1].set_title(r"$\dot{y}$", fontsize=20)
        axs[0, 1].set_ylabel(r"Joint velocity [$deg/s$]", fontsize=14)
        axs[0, 1].set_xlabel("Time [s]", fontsize=14)
        axs[0, 1].legend(fontsize=12)
        axs[0, 2].plot(time, dmp.y[:, i, 2], label="output")
        axs[0, 2].plot(time, dmp.T[:, i, 2], label="regression")
        axs[0, 2].set_title(r"$\ddot{y}$", fontsize=20)
        axs[0, 2].set_ylabel(r"Joint acceleration [$deg/s^2$]", fontsize=14)
        axs[0, 2].set_xlabel("Time [s]", fontsize=14)
        axs[0, 2].legend(fontsize=12)

        # plot internal states
        axs[1, 0].plot(time, dmp.z_history[:, i, 0])
        axs[1, 0].set_title("z", fontsize=20)
        axs[1, 0].set_xlabel("Time [s]", fontsize=14)
        axs[1, 1].plot(time, dmp.z_history[:, i, 1])
        axs[1, 1].set_title(r"$\dot{z}$", fontsize=20)
        axs[1, 1].set_xlabel("Time [s]", fontsize=14)
        axs[1, 2].plot(time, dmp.psi_history[:, i, :])
        axs[1, 2].set_title("Weighting Kernels", fontsize=20)
        axs[1, 2].set_xlabel("Time [s]", fontsize=14)

        axs[2, 0].plot(time, dmp.x_history[:, i, 0])
        axs[2, 0].set_title("x", fontsize=20)
        axs[2, 0].set_xlabel("Time [s]", fontsize=14)
        axs[2, 1].plot(time, dmp.x_history[:, i, 1])
        axs[2, 1].set_title(r"$\dot{x}$", fontsize=20)
        axs[2, 1].set_xlabel("Time [s]", fontsize=14)
        axs[2, 2].plot(dmp.w_history[-1, i, :])
        axs[2, 2].set_title("Weights", fontsize=20)
        axs[2, 2].set_xlabel("Number rbfs", fontsize=14)
        # set the spacing between subplots
        plt.subplots_adjust(
            left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5
        )
    plt.show()
