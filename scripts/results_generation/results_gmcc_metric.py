import json
import logging
import statistics
from pathlib import Path

import numpy as np

# from team.dmp_trajectory import DmpTrajectory
from team.dynamical_movement_primitives import DynamicMovementPrimitives
from team.gofa_trajectory import GoFaTrajectory
from team.utility.accuracy_metric import symmetric_gmcc


def first_is_longer(traj_1: np.ndarray, traj_2: np.ndarray) -> bool:
    """
    Assess if first trajectory is longer than second one

    :param traj_1: first trajectory to compare
    :param traj_2: second trajectory to compare
    :return: if first trajectory is longer than second one
    """
    return np.shape(traj_1)[0] > np.shape(traj_2)[0]


def get_motion_onset_idx(trajectory: np.ndarray, tol: float) -> int:
    """
    Get index at which motion onset occurs

    :param trajectory: data
    :param tol: tolerance to assess motion onset
    :return: index at which motion starts
    """
    nb_samples = np.shape(trajectory)[0]
    dims = np.shape(trajectory)[1]
    list_idx = []
    # Loop on each dimension
    for i in range(dims):
        j = 0
        while abs(trajectory[j, i] - trajectory[j + 1, i]) < tol:
            j = j + 1
            if j == nb_samples - 1:
                break
        list_idx.append(j)
    # Take lowest index as some dimensions might not be affected by motion
    return min(list_idx)


def pad_end(trajectory: np.ndarray, final_len: int) -> np.ndarray:
    """
    Pad end of trajectory

    :param trajectory: data
    :param final_len: final trajectory length
    :return: padded trajectory
    """
    nb_samples = final_len - np.shape(trajectory)[0]
    nb_dims = np.shape(trajectory)[1]
    extension = np.full_like(trajectory, trajectory[-1, :], shape=(nb_samples, nb_dims))
    return np.row_stack([trajectory, extension])


def comparison_setup(
    regression: np.ndarray, reproduction: np.ndarray
) -> list[np.ndarray]:
    """
    Manipulate regression and reproduction trajectories to bring them at same length
    1. Trim initial static portion of both trajectories, not moving data
    2. Check which trajectory is longer
    3. Pad shorter trajectory to same length

    :param regression: regression trajectory
    :param reproduction: reproduction trajectory
    :return: list with processed trajectories ready to be compared
    """
    idx_regre = get_motion_onset_idx(regression, tol=10 ** (-1))
    idx_repro = get_motion_onset_idx(reproduction, tol=10 ** (-1))
    trimmed_regression = regression[idx_regre:, :]
    trimmed_reproduction = reproduction[idx_repro:, :]
    if first_is_longer(trimmed_regression, trimmed_reproduction):
        final_len = np.shape(trimmed_regression)[0]
        trimmed_reproduction = pad_end(trimmed_reproduction, final_len)
    else:
        final_len = np.shape(trimmed_reproduction)[0]
        trimmed_regression = pad_end(trimmed_regression, final_len)
    return [trimmed_regression, trimmed_reproduction]


def get_gmcc_score_folder(data_path: Path) -> tuple[float, float]:
    """Get the gmcc mean and standard deviation of a task

    :param data_path: a path to the data
    :return: mean and standard deviation of gmcc
    """

    logging.basicConfig(force=True)
    team_logger = logging.getLogger("team")
    team_logger.setLevel(logging.INFO)

    dmp = DynamicMovementPrimitives.load_dmp(data_path)

    folders = [f for f in data_path.iterdir() if f.is_dir()]
    gmccs = []
    team_logger.info("Calculating gmccs for " + str(data_path))
    for folder in folders:
        if folder.stem == "demonstrations":
            continue

        if Path(folder, "gmcc.json").exists():
            team_logger.info("Loading gmcc for " + str(folder.stem))
            with open(Path(folder, "gmcc.json"), "rb") as f:
                json_data = json.load(f)
                gmccs.append(json_data["gmcc"])
            continue

        team_logger.info("Calculating gmcc for " + str(folder.stem))
        with open(Path(folder, "task_config.json"), "rb") as f:
            json_data = json.load(f)
        trajectory = GoFaTrajectory(np.array(json_data["trajectory"]))
        regre, repro = comparison_setup(dmp.regression[:, 1:], trajectory.joints)

        # IN case we need TCP
        # trajectory_tcp = GoFaTrajectory(np.array(json_data["trajectory"]))
        # regression_tcp_traj = GoFaTrajectory(dmp.regression[:, 1:])
        # regre, repro = comparison_setup(regression_tcp_traj.tcp, trajectory_tcp.tcp)

        gmcc = symmetric_gmcc(regre, repro)
        with open(Path(folder, "gmcc.json"), "w") as f:
            json.dump({"gmcc": gmcc}, f)
        gmccs.append(gmcc)

    mean_gmcc = statistics.mean(gmccs)
    std_gmcc = statistics.stdev(gmccs)
    with open(Path(data_path, "gmccs.json"), "w") as f:
        json.dump({"gmccs": mean_gmcc, "std": std_gmcc}, f)
    return mean_gmcc, std_gmcc


if __name__ == "__main__":
    root_folder = Path(__file__).parent.parent.parent.resolve()
    data_dir = Path(root_folder, "data", "Ebikon_07_22_2022", "Ebikon_07_22_2022")
    folders = [f for f in data_dir.iterdir() if f.is_dir()]
    gmccs = []
    for folder in folders:
        gmcc, std_dev = get_gmcc_score_folder(data_path=folder)
        gmccs.append(gmcc)

    # data_dir = os.path.join(
    #     os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    #     "data/reproductions",
    # )

    # # Brush homing score
    # regression_path = os.path.join(data_dir, "brush_homing/tcp_regression.npy")
    # reproduction_path = os.path.join(data_dir, "brush_homing/brush_homing_repro.json")
    # gmcc_value_bh = get_gmcc_score(regression_path, reproduction_path, "tcp")
    # # Brush picking up score
    # regression_path = os.path.join(data_dir, "brush_picking_up/tcp_regression.npy")
    # reproduction_path = os.path.join(
    #     data_dir, "brush_picking_up/brush_picking_up_repro.json"
    # )
    # gmcc_value_pu = get_gmcc_score(regression_path, reproduction_path, "tcp")
    # # Door closing score
    # regression_path = os.path.join(data_dir, "door_closing/tcp_regression.npy")
    # reproduction_path = os.path.join(data_dir, "door_closing/door_closing_repro.json")
    # gmcc_value_dc = get_gmcc_score(regression_path, reproduction_path, "tcp")
    # # Door opening score
    # regression_path = os.path.join(data_dir, "door_opening/tcp_regression.npy")
    # reproduction_path = os.path.join(data_dir, "door_opening/door_opening_repro.json")
    # gmcc_value_do = get_gmcc_score(regression_path, reproduction_path, "tcp")
    # # Rail cleaning score
    # regression_path = os.path.join(data_dir, "rail_cleaning/tcp_regression.npy")
    # reproduction_path = os.path.join(data_dir, "rail_cleaning/rail_cleaning_repro.json")
    # gmcc_value_rc = get_gmcc_score(regression_path, reproduction_path, "tcp")

    # # Drill task score
    # regression_path = os.path.join(data_dir, "drill/regression/demonstration_1.json")
    # reproduction_path = os.path.join(data_dir, "drill/drill_repro/demonstration_1.json")
    # gmcc_value_drill = get_gmcc_score(regression_path, reproduction_path, "tcp")
    # # Drill far task score
    # regression_path = os.path.join(data_dir, "drill/regression/demonstration_1.json")
    # reproduction_path = os.path.join(data_dir, "drill/drill_far/demonstration_1.json")
    # gmcc_value_dr_f = get_gmcc_score(regression_path, reproduction_path, "tcp")
    # # Timo avoid task score
    # regression_path = os.path.join(
    #     data_dir, "Timo_avoid/regression/demonstration_1.json"
    # )
    # reproduction_path = os.path.join(data_dir, "Timo_avoid/repro/demonstration_1.json")
    # gmcc_value_ta = get_gmcc_score(regression_path, reproduction_path, "tcp")
    # # Timo avoid far task score
    # regression_path = os.path.join(
    #     data_dir, "Timo_avoid/regression/demonstration_1.json"
    # )
    # reproduction_path = os.path.join(
    #     data_dir, "Timo_avoid/repro_far/demonstration_1.json"
    # )
    # gmcc_value_ta_f = get_gmcc_score(regression_path, reproduction_path, "tcp")
    # # Timo collab task score
    # regression_path = os.path.join(
    #     data_dir, "Timo_collab/regression/demonstration_1.json"
    # )
    # reproduction_path = os.path.join(data_dir, "Timo_collab/repro/demonstration_1.json")
    # gmcc_value_tc = get_gmcc_score(regression_path, reproduction_path, "tcp")
    # # Timo collab far task score
    # regression_path = os.path.join(
    #     data_dir, "Timo_collab/regression/demonstration_1.json"
    # )
    # reproduction_path = os.path.join(
    #     data_dir, "Timo_collab/repro_far/demonstration_1.json"
    # )
    # gmcc_value_tc_f = get_gmcc_score(regression_path, reproduction_path, "tcp")
    # # Timo difficult task score
    # regression_path = os.path.join(
    #     data_dir, "Timo_difficult/regression/demonstration_1.json"
    # )
    # reproduction_path = os.path.join(
    #     data_dir, "Timo_difficult/repro/demonstration_1.json"
    # )
    # gmcc_value_td = get_gmcc_score(regression_path, reproduction_path, "tcp")
    # # Timo difficult far task score
    # regression_path = os.path.join(
    #     data_dir, "Timo_difficult/regression/demonstration_1.json"
    # )
    # reproduction_path = os.path.join(
    #     data_dir, "Timo_difficult/repro_far/demonstration_1.json"
    # )
    # gmcc_value_td_f = get_gmcc_score(regression_path, reproduction_path, "tcp")
    # # Drill data
    # drill = [gmcc_value_drill, gmcc_value_dr_f]
    # # Avoid data
    # avoid = [gmcc_value_ta, gmcc_value_ta_f]
    # # Parallel
    # parallel = [gmcc_value_td, gmcc_value_td_f]
    # # Collab
    # collab = [gmcc_value_tc, gmcc_value_tc_f]

    # text_file = (
    #     r"\begin{tabular}{l  c  c  c  c } "
    #     "\n "
    #     r"\toprule "
    #     "\n "
    #     r"Task & F1 & F2 & F3 & F4 \\ [0.5ex] "
    #     "\n "
    #     r"\midrule "
    #     "\n "
    #     r"\makecell[l]{Name} & \makecell{Drill} & \makecell{Avoid} & "
    #     r"\makecell{Parallel} & \makecell{Collaborative} "
    #     r"\\ "
    #     "\n"
    #     r" \addlinespace[0.5em] "
    #     " \n "
    #     r"\makecell[l]{Average demonstration\\ duration [s]} "
    #     f"& ${np.around(np.mean(drill), 3)}"
    #     r"\pm"
    #     f"{np.around(np.std(drill), 3)}$ &"
    #     f" ${np.around(np.mean(avoid), 3)}"
    #     r"\pm"
    #     f"{np.around(np.std(avoid), 3)}$ &"
    #     f" ${np.around(np.mean(parallel), 3)}"
    #     r"\pm"
    #     f"{np.around(np.std(parallel), 3)}$ &"
    #     f" ${np.around(np.mean(collab), 3)}"
    #     r"\pm"
    #     f"{np.around(np.std(collab), 3)}$"
    #     r"\\ "
    #     "\n "
    #     r" \bottomrule "
    #     "\n "
    #     r"\end{tabular}"
    # )

    # with open("gmcc_results.tex", "w") as f:
    #     f.write(text_file)
