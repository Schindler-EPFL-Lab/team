import time

from team.aligned_trajectories import AlignedTrajectories
from team.dynamical_movement_primitives import DynamicMovementPrimitives
from team.gaussian_mixture_regression import GMR
from team.probabilistic_encoding import ProbabilisticEncoding
from team.trajectory import Trajectory
from team.utility.handling_data import get_demo_files


def compute_average_duration(data_dir: str) -> list:

    files_paths = get_demo_files(data_dir)
    average_dur = []
    for file_path in files_paths:
        traj = Trajectory.from_file(file_path)
        average_dur.append(traj.timestamps[-1])
    return average_dur


def time_breakdown(data_dir: str, runs: int) -> tuple[list, list, list, list, list]:

    prep_time = []
    encod_time = []
    bo_time = []
    traj_time = []
    total_time = []
    for i in range(runs):
        t_start = time.time()
        trajectories = AlignedTrajectories.load_dataset_and_preprocess(data_dir)
        prep_time.append(time.time() - t_start)
        t_start_2 = time.time()
        pe = ProbabilisticEncoding(
            trajectories, max_nb_components=10, min_nb_components=3, iterations=10
        )
        regression = GMR(trajectories, pe)
        encod_time.append(time.time() - t_start_2)
        target = regression.prediction[-1, 1:]
        initial_state = regression.prediction[0, 1:]
        t_start_3 = time.time()
        dmp = DynamicMovementPrimitives(
            regression=regression.prediction,
            c_order=1,
            initial_joints=initial_state,
            goal_joints=target,
        )
        bo_time.append(time.time() - t_start_3)
        t_start_4 = time.time()
        _ = dmp.compute_joint_dynamics(goal=target, y_init=initial_state)
        traj_time.append(time.time() - t_start_4)
        total_time.append(time.time() - t_start)

    return prep_time, encod_time, bo_time, traj_time, total_time
