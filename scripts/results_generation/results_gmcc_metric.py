import os

import numpy as np
import pandas as pd

from team.utility.accuracy_metric import symmetric_gmcc


def extract_tcp_into_numpy(data_path: str) -> np.ndarray:
    """
    Extract tcp data from json dictionary into numpy array

    :param data_path: data path
    :return: extracted data in numpy array
    """
    df = pd.read_json(data_path)
    tcp = df[["tcp_x", "tcp_y", "tcp_z"]]
    return tcp.to_numpy()


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
    dims = np.shape(trajectory)[1]
    list_idx = []
    # Loop on each dimension
    for i in range(dims):
        j = 0
        while (abs(trajectory[j, i] - trajectory[j+1, i]) < tol):
            j = j+1
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
    extension = np.full_like(trajectory, trajectory[-1, :], shape=(nb_samples, 3))
    return np.row_stack([trajectory, extension])


def comparison_setup(regression: np.ndarray, reproduction: np.ndarray) -> list[
        np.ndarray]:
    """
    Manipulate regression and reproduction trajectories to bring them at same length
    1. Trim initial static portion of both trajectories, not moving data
    2. Check which trajectory is longer
    3. Pad shorter trajectory to same length

    :param regression: regression trajectory
    :param reproduction: reproduction trajectory
    :return: list with processed trajectories ready to be compared
    """
    idx_regre = get_motion_onset_idx(regression, tol=10**(-1))
    idx_repro = get_motion_onset_idx(reproduction, tol=10**(-1))
    trimmed_regression = regression[idx_regre:, :]
    trimmed_reproduction = reproduction[idx_repro:, :]
    if first_is_longer(trimmed_regression, trimmed_reproduction):
        final_len = np.shape(trimmed_regression)[0]
        trimmed_reproduction = pad_end(trimmed_reproduction, final_len)
    else:
        final_len = np.shape(trimmed_reproduction)[0]
        trimmed_regression = pad_end(trimmed_regression, final_len)
    return [trimmed_regression, trimmed_reproduction]


def get_gmcc_score(regression_path: str, reproduction_path: str) -> float:
    """_summary_

    :param regression_path: regression data path
    :param reproduction_path: reproduction data path
    :return: gmcc score
    """
    file_extension = os.path.splitext(regression_path)[1]
    if file_extension == ".json":
        regression = extract_tcp_into_numpy(regression_path)
    else:
        # Account for regression files in .npy
        regression = np.load(regression_path)
    reproduction = extract_tcp_into_numpy(reproduction_path)
    (regre, repro) = comparison_setup(regression, reproduction)
    return symmetric_gmcc(regre, repro)


if __name__ == "__main__":

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data/reproductions",
    )
    # Brush homing score
    regression_path = os.path.join(data_dir, "brush_homing/tcp_regression.npy")
    reproduction_path = os.path.join(data_dir, "brush_homing/brush_homing_repro.json")
    gmcc_value_bh = get_gmcc_score(regression_path, reproduction_path)
    # Brush picking up score
    regression_path = os.path.join(data_dir, "brush_picking_up/tcp_regression.npy")
    reproduction_path = os.path.join(data_dir,
                                     "brush_picking_up/brush_picking_up_repro.json")
    gmcc_value_pu = get_gmcc_score(regression_path, reproduction_path)
    # Door closing score
    regression_path = os.path.join(data_dir, "door_closing/tcp_regression.npy")
    reproduction_path = os.path.join(data_dir,
                                     "door_closing/door_closing_repro.json")
    gmcc_value_dc = get_gmcc_score(regression_path, reproduction_path)
    # Door opening score
    regression_path = os.path.join(data_dir, "door_opening/tcp_regression.npy")
    reproduction_path = os.path.join(data_dir,
                                     "door_opening/door_opening_repro.json")
    gmcc_value_do = get_gmcc_score(regression_path, reproduction_path)
    # Rail cleaning score
    regression_path = os.path.join(data_dir, "rail_cleaning/tcp_regression.npy")
    reproduction_path = os.path.join(data_dir,
                                     "rail_cleaning/rail_cleaning_repro.json")
    gmcc_value_rc = get_gmcc_score(regression_path, reproduction_path)
    # Drill task score
    regression_path = os.path.join(data_dir, "drill/regression/demonstration_1.json")
    reproduction_path = os.path.join(data_dir, "drill/drill_repro/demonstration_1.json")
    gmcc_value_drill = get_gmcc_score(regression_path, reproduction_path)
    # Drill far task score
    regression_path = os.path.join(data_dir, "drill/regression/demonstration_1.json")
    reproduction_path = os.path.join(data_dir, "drill/drill_far/demonstration_1.json")
    gmcc_value_dr_f = get_gmcc_score(regression_path, reproduction_path)
    # Timo avoid task score
    regression_path = os.path.join(data_dir,
                                   "Timo_avoid/regression/demonstration_1.json")
    reproduction_path = os.path.join(data_dir,
                                     "Timo_avoid/repro/demonstration_1.json")
    gmcc_value_ta = get_gmcc_score(regression_path, reproduction_path)
    # Timo avoid far task score
    regression_path = os.path.join(data_dir,
                                   "Timo_avoid/regression/demonstration_1.json")
    reproduction_path = os.path.join(data_dir,
                                     "Timo_avoid/repro_far/demonstration_1.json")
    gmcc_value_ta_f = get_gmcc_score(regression_path, reproduction_path)
    # Timo collab task score
    regression_path = os.path.join(data_dir,
                                   "Timo_collab/regression/demonstration_1.json")
    reproduction_path = os.path.join(data_dir,
                                     "Timo_collab/repro/demonstration_1.json")
    gmcc_value_tc = get_gmcc_score(regression_path, reproduction_path)
    # Timo collab far task score
    regression_path = os.path.join(data_dir,
                                   "Timo_collab/regression/demonstration_1.json")
    reproduction_path = os.path.join(data_dir,
                                     "Timo_collab/repro_far/demonstration_1.json")
    gmcc_value_tc_f = get_gmcc_score(regression_path, reproduction_path)
    # Timo difficult task score
    regression_path = os.path.join(data_dir,
                                   "Timo_difficult/regression/demonstration_1.json")
    reproduction_path = os.path.join(data_dir,
                                     "Timo_difficult/repro/demonstration_1.json")
    gmcc_value_td = get_gmcc_score(regression_path, reproduction_path)
    # Timo difficult far task score
    regression_path = os.path.join(data_dir,
                                   "Timo_difficult/regression/demonstration_1.json")
    reproduction_path = os.path.join(data_dir,
                                     "Timo_difficult/repro_far/demonstration_1.json")
    gmcc_value_td_f = get_gmcc_score(regression_path, reproduction_path)
    # Drill data
    drill = [gmcc_value_drill, gmcc_value_dr_f]
    # Avoid data
    avoid = [gmcc_value_ta, gmcc_value_ta_f]
    # Parallel
    parallel = [gmcc_value_td, gmcc_value_td_f]
    # Collab
    collab = [gmcc_value_tc, gmcc_value_tc_f]

    text_file = (
        r"\begin{tabular}{l  c  c  c  c } "
        "\n "
        r"\toprule "
        "\n "
        r"Task & F1 & F2 & F3 & F4 \\ [0.5ex] "
        "\n "
        r"\midrule "
        "\n "
        r"\makecell[l]{Name} & \makecell{Drill} & \makecell{Avoid} & "
        r"\makecell{Parallel} & \makecell{Collaborative} "
        r"\\ "
        "\n"
        r" \addlinespace[0.5em] "
        " \n "
        r"\makecell[l]{Average demonstration\\ duration [s]} "
        f"& ${np.around(np.mean(drill), 3)}"
        r"\pm"
        f"{np.around(np.std(drill), 3)}$ &"
        f" ${np.around(np.mean(avoid), 3)}"
        r"\pm"
        f"{np.around(np.std(avoid), 3)}$ &"
        f" ${np.around(np.mean(parallel), 3)}"
        r"\pm"
        f"{np.around(np.std(parallel), 3)}$ &"
        f" ${np.around(np.mean(collab), 3)}"
        r"\pm"
        f"{np.around(np.std(collab), 3)}$"
        r"\\ "
        "\n "
        r" \bottomrule "
        "\n "
        r"\end{tabular}"
    )

    with open("gmcc_results.tex", "w") as f:
        f.write(text_file)
