import json
import logging
import statistics
from pathlib import Path

import numpy as np

from team.dmp_trajectory import DmpTrajectory
from team.dynamical_movement_primitives import DynamicMovementPrimitives
from team.trajectory_base import TrajectoryBase


def comparison_setup(regression: TrajectoryBase, reproduction: TrajectoryBase) -> None:
    """
    Manipulate regression and reproduction trajectories to bring them at same length
    1. Trim initial static portion of both trajectories, not moving data
    2. Check which trajectory is longer
    3. Pad shorter trajectory to same length

    :param regression: regression trajectory
    :param reproduction: reproduction trajectory
    :return: list with processed trajectories ready to be compared
    """

    if regression.is_longer(reproduction):
        final_len = len(regression)
        reproduction.pad_end_to(final_len)
        return

    final_len = len(reproduction)
    regression.pad_end_to(final_len)


def get_score_folder(
    data_path: Path,
    filename: str,
    rewrite_gmcc: bool = False,
) -> tuple[float, float, float, float]:
    """Get the gmcc mean and joint errors (with std for both) of a task

    :param data_path: a path to the data
    :param filename: filename of reproductions
    :param rewrite_gmcc: if True the gmcc is recalculated, if False, if it was
    calculated before it is not recalculated. default to False because gmcc calculation
    is slow
    :return: mean and standard deviation of gmcc, mean and std of joint errror
    """

    logging.basicConfig(force=True)
    team_logger = logging.getLogger("team")
    team_logger.setLevel(logging.DEBUG)

    dmp = DynamicMovementPrimitives.load(data_path)

    folders = [f for f in data_path.iterdir() if f.is_dir()]
    gmccs = []
    joint_mean_abs_errors = []
    team_logger.info("Calculating gmccs for " + str(data_path))
    for folder in folders:
        if "reproduction" not in folder.stem:
            continue

        reproduction, _, target = DmpTrajectory.from_config_file(Path(folder, filename))
        joint_mean_abs_error = (
            np.absolute(reproduction.get_joints_at_index(-1) - target).sum()
            / reproduction.nb_of_joints
        )

        if not rewrite_gmcc and Path(folder, "scores.json").exists():
            team_logger.info("Loading gmcc for " + str(folder.stem))
            with open(Path(folder, "scores.json"), "rb") as f:
                json_data = json.load(f)
                gmcc = json_data["gmcc"]
        else:
            team_logger.info("Calculating gmcc for " + str(folder.stem))
            comparison_setup(dmp.regression, reproduction)
            gmcc = reproduction.symmetric_gmcc(dmp.regression)

        with open(Path(folder, "scores.json"), "w") as f:
            json.dump(
                {"gmcc": gmcc, "joints mean absolute error": joint_mean_abs_error}, f
            )
        gmccs.append(gmcc)
        joint_mean_abs_errors.append(joint_mean_abs_error)

    mean_gmcc = statistics.mean(gmccs)
    std_gmcc = statistics.stdev(gmccs)
    mean_joint_errors = statistics.mean(joint_mean_abs_errors)
    std_joint_errors = statistics.stdev(joint_mean_abs_errors)

    with open(Path(data_path, "scores.json"), "w") as f:
        json.dump(
            {
                "gmccs": {"mean": mean_gmcc, "std": std_gmcc},
                "joints mean absolute error": {
                    "mean": mean_joint_errors,
                    "std": std_joint_errors,
                },
            },
            f,
        )
    return mean_gmcc, std_gmcc, mean_joint_errors, std_joint_errors
