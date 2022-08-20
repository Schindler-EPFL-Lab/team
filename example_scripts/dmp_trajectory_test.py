import os
from pathlib import Path

import numpy as np
from rws2.RWS2 import RWS

from team.aligned_trajectories import AlignedTrajectories
from team.dynamical_movement_primitives import DynamicMovementPrimitives
from team.gaussian_mixture_regression import GMR
from team.probabilistic_encoding import ProbabilisticEncoding
from team.utility.dmp_visualization import plotting
from team.utility.gmm_visualization import plot_gmm, plot_js_distance

if __name__ == "__main__":

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "demonstrations/right_rail_cleaning",
    )
    trajectories = AlignedTrajectories.load_dataset_and_preprocess(data_dir)
    pe = ProbabilisticEncoding(
        trajectories, max_nb_components=10, min_nb_components=4, iterations=20
    )
    plot_gmm(pe)
    plot_js_distance(pe, max_nb_components=10, min_nb_components=4)
    regression = GMR(trajectories, pe)
    # retrieve target goal (goal from camera, initial joints from robot)
    target = np.array([-58, 60, -41, -79, -61, 67.7])
    initial_state = np.array([-41, 24, 24, -51, -60, 31.16])
    dmp = DynamicMovementPrimitives(
        regression=regression.prediction,
        c_order=1,
        initial_joints=initial_state,
        goal_joints=target,
    )
    dmp_traj = dmp.compute_joint_dynamics(goal=target, y_init=initial_state)
    plotting(dmp)
    store_path = Path.cwd().parent.joinpath("models/right_rail_cleaning")
    dmp.save_dmp(dir_path=store_path)
    text = dmp_traj.joints_to_string()
    rws = RWS("https://localhost:8881")
    rws.upload_text_file_to_controller(
        text_data=text, filename="right_rail_cleaning.txt"
    )
