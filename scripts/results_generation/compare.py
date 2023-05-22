import json
import logging
import statistics
from pathlib import Path

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


def get_gmcc_score_folder(
    data_path: Path, rewrite: bool, filename: str
) -> tuple[float, float]:
    """Get the gmcc mean and standard deviation of a task

    :param data_path: a path to the data
    :param rewrite:
    :param filename:
    :return: mean and standard deviation of gmcc
    """

    logging.basicConfig(force=True)
    team_logger = logging.getLogger("team")
    team_logger.setLevel(logging.DEBUG)

    dmp = DynamicMovementPrimitives.load_dmp(data_path)

    folders = [f for f in data_path.iterdir() if f.is_dir()]
    gmccs = []
    team_logger.info("Calculating gmccs for " + str(data_path))
    for folder in folders:
        if "reproduction" not in folder.stem:
            continue

        if not rewrite and Path(folder, "gmcc.json").exists():
            team_logger.info("Loading gmcc for " + str(folder.stem))
            with open(Path(folder, "gmcc.json"), "rb") as f:
                json_data = json.load(f)
                gmccs.append(json_data["gmcc"])
            continue

        team_logger.info("Calculating gmcc for " + str(folder.stem))

        reproduction = DmpTrajectory.from_config_file(Path(folder, filename))
        comparison_setup(dmp.regression, reproduction)

        gmcc = reproduction.symmetric_gmcc(dmp.regression)

        with open(Path(folder, "gmcc.json"), "w") as f:
            json.dump({"gmcc": gmcc}, f)
        gmccs.append(gmcc)

    mean_gmcc = statistics.mean(gmccs)
    std_gmcc = statistics.stdev(gmccs)
    with open(Path(data_path, "gmccs.json"), "w") as f:
        json.dump({"gmccs": mean_gmcc, "std": std_gmcc}, f)
    return mean_gmcc, std_gmcc
