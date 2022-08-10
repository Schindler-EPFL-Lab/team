import os

import numpy as np

from results_generation_utils import compute_average_duration, time_breakdown


if __name__ == "__main__":

    drill_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "arco-datasets/arco_control/Ebikon_07_21_2022/Ebikon_07_21_2022/drill/"
        "demonstrations",
    )
    drill_average_dur = compute_average_duration(drill_data_dir)
    (
        drill_prep_time,
        drill_encod_time,
        drill_bo_time,
        drill_traj_time,
        drill_total_time,
    ) = time_breakdown(drill_data_dir, runs=30)

    avoid_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "arco-datasets/arco_control/Ebikon_07_21_2022/Ebikon_07_21_2022/Timo_avoid/"
        "demonstrations",
    )
    avoid_average_dur = compute_average_duration(avoid_data_dir)
    (
        avoid_prep_time,
        avoid_encod_time,
        avoid_bo_time,
        avoid_traj_time,
        avoid_total_time,
    ) = time_breakdown(avoid_data_dir, runs=30)

    diff_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "arco-datasets/arco_control/Ebikon_07_21_2022/Ebikon_07_21_2022/TIMO_difficult/"
        "demonstrations",
    )
    diff_average_dur = compute_average_duration(diff_data_dir)
    (
        diff_prep_time,
        diff_encod_time,
        diff_bo_time,
        diff_traj_time,
        diff_total_time,
    ) = time_breakdown(diff_data_dir, runs=30)

    collab_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "arco-datasets/arco_control/Ebikon_07_21_2022/Ebikon_07_21_2022/TIMO_collab/"
        "demonstrations",
    )
    collab_average_dur = compute_average_duration(collab_data_dir)
    (
        collab_prep_time,
        collab_encod_time,
        collab_bo_time,
        collab_traj_time,
        collab_total_time,
    ) = time_breakdown(collab_data_dir, runs=30)

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
        "\n "
        r"\makecell[l]{Nb demonstrations}  & 3 & 3 & 3 & 3"
        r"\\ "
        "\n "
        r"\addlinespace[0.5em]"
        " \n "
        r"\makecell[l]{Average demonstration\\ duration [s]} "
        f"& ${np.around(np.mean(drill_average_dur), 3)}"
        r"\pm"
        f"{np.around(np.std(drill_average_dur), 3)}$ &"
        f" ${np.around(np.mean(avoid_average_dur), 3)}"
        r"\pm"
        f"{np.around(np.std(avoid_average_dur), 3)}$ &"
        f" ${np.around(np.mean(diff_average_dur), 3)}"
        r"\pm"
        f"{np.around(np.std(diff_average_dur), 3)}$ &"
        f" ${np.around(np.mean(collab_average_dur), 3)}"
        r"\pm"
        f"{np.around(np.std(collab_average_dur), 3)}$"
        r"\\ "
        "\n "
        r"\addlinespace[0.5em] "
        "\n "
        r"\makecell[l]{DTW [s]} & "
        f"${np.around(np.mean(drill_prep_time), 3)}"
        r"\pm"
        f"{np.around(np.std(drill_prep_time), 3)}$ &"
        f"${np.around(np.mean(avoid_prep_time), 3)}"
        r"\pm"
        f"{np.around(np.std(avoid_prep_time), 3)}$ & "
        f"${np.around(np.mean(diff_prep_time), 3)}"
        r"\pm"
        f"{np.around(np.std(diff_prep_time), 3)}$ & "
        f"${np.around(np.mean(collab_prep_time), 3)}"
        r"\pm"
        f"{np.around(np.std(collab_prep_time), 3)}$  "
        r"\\ "
        "\n "
        r"\addlinespace[0.5em] "
        "\n "
        r"\makecell[l]{GMM + GMR [s]} &"
        f" ${np.around(np.mean(drill_encod_time), 3)}"
        r"\pm"
        f"{np.around(np.std(drill_encod_time), 3)}$ & "
        f"${np.around(np.mean(avoid_encod_time), 3)}"
        r"\pm"
        f"{np.around(np.std(avoid_encod_time), 3)}$ &"
        f" ${np.around(np.mean(diff_encod_time), 3)}"
        r"\pm"
        f"{np.around(np.std(diff_encod_time), 3)}$ & "
        f"${np.around(np.mean(collab_encod_time), 3)}"
        r"\pm"
        f"{np.around(np.std(collab_encod_time), 3)}$  "
        r"\\ "
        "\n "
        r"\addlinespace[0.5em] "
        "\n"
        r" \makecell[l]{BO [s]} & "
        f"${np.around(np.mean(drill_bo_time), 3)}"
        r"\pm"
        f"{np.around(np.std(drill_bo_time), 3)}$ & "
        f"${np.around(np.mean(avoid_bo_time), 3)}"
        r"\pm"
        f"{np.around(np.std(avoid_bo_time), 3)}$ & "
        f" ${np.around(np.mean(diff_bo_time), 3)}"
        r"\pm"
        f"{np.around(np.std(diff_bo_time), 3)}$ &"
        f" ${np.around(np.mean(collab_bo_time), 3)}"
        r"\pm"
        f"{np.around(np.std(collab_bo_time), 3)}$ "
        r"\\ "
        "\n"
        r" \addlinespace[0.5em] "
        "\n "
        r" \makecell[l]{ProMP[s]} & "
        f" ${np.around(np.mean(drill_traj_time), 3)}"
        r"\pm"
        f"{np.around(np.std(drill_traj_time), 3)}$ & "
        f"${np.around(np.mean(avoid_traj_time), 3)}"
        r"\pm"
        f"{np.around(np.std(avoid_traj_time), 3)}$ & "
        f"${np.around(np.mean(diff_traj_time), 3)}"
        r"\pm"
        f"{np.around(np.std(diff_traj_time), 3)}$ & "
        f"${np.around(np.mean(collab_traj_time), 3)}"
        r"\pm"
        f"{np.around(np.std(collab_traj_time), 3)}$  "
        r"\\ "
        "\n"
        r" \addlinespace[0.5em] "
        "\n"
        r"\makecell[l]{Total time [s]} "
        f"& ${np.around(np.mean(drill_total_time), 3)}"
        r"\pm"
        f"{np.around(np.std(drill_total_time), 3)}$ &"
        f" ${np.around(np.mean(avoid_total_time), 3)}"
        r"\pm"
        f"{np.around(np.std(avoid_total_time), 3)}$ &"
        f" ${np.around(np.mean(diff_total_time), 3)}"
        r"\pm"
        f"{np.around(np.std(diff_total_time), 3)}$ &"
        f" ${np.around(np.mean(collab_total_time), 3)}"
        r"\pm"
        f"{np.around(np.std(collab_total_time), 3)}$ "
        r"\\ "
        "\n"
        r" \bottomrule "
        "\n "
        r"\end{tabular}"
    )

    with open("factory_tasks.tex", "w") as f:
        f.write(text_file)
