import os

import numpy as np

from results_generation_utils import compute_average_duration, time_breakdown

if __name__ == "__main__":

    bpu_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "arco-datasets/arco_control/learning_from_demonstrations/brush_picking_up",
    )
    bpu_average_dur = compute_average_duration(bpu_data_dir)
    (
        bpu_prep_time,
        bpu_encod_time,
        bpu_bo_time,
        bpu_traj_time,
        bpu_total_time,
    ) = time_breakdown(bpu_data_dir, runs=10)

    do_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "arco-datasets/arco_control/learning_from_demonstrations/door_opening",
    )
    do_average_dur = compute_average_duration(do_data_dir)
    (
        do_prep_time,
        do_encod_time,
        do_bo_time,
        do_traj_time,
        do_total_time,
    ) = time_breakdown(do_data_dir, runs=10)

    cr_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "arco-datasets/arco_control/learning_from_demonstrations/rail_cleaning",
    )
    cr_average_dur = compute_average_duration(cr_data_dir)
    (
        cr_prep_time,
        cr_encod_time,
        cr_bo_time,
        cr_traj_time,
        cr_total_time,
    ) = time_breakdown(cr_data_dir, runs=10)

    db_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "arco-datasets/arco_control/learning_from_demonstrations/brush_homing",
    )
    db_average_dur = compute_average_duration(db_data_dir)
    (
        db_prep_time,
        db_encod_time,
        db_bo_time,
        db_traj_time,
        db_total_time,
    ) = time_breakdown(db_data_dir, runs=10)

    dc_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "arco-datasets/arco_control/learning_from_demonstrations/door_closing",
    )
    dc_average_dur = compute_average_duration(dc_data_dir)
    (
        dc_prep_time,
        dc_encod_time,
        dc_bo_time,
        dc_traj_time,
        dc_total_time,
    ) = time_breakdown(dc_data_dir, runs=10)

    text_file = (
        r"\begin{tabular}{l  c  c  c  c  c} "
        "\n "
        r"\toprule "
        "\n "
        r"Task & T1 & T2 & T3 & T4 & T5 \\ [0.5ex] "
        "\n "
        r"\midrule "
        "\n "
        r"\makecell[l]{Name} & \makecell{Open the\\ door} & \makecell{Pick up\\ brush}"
        r"& \makecell{Clean\\ rail} & \makecell{Deposit\\ brush} "
        r"& \makecell{Close\\ the door} "
        r"\\ "
        "\n"
        r" \addlinespace[0.5em] "
        "\n "
        r"\makecell[l]{Nb demonstrations}  & 6 & 4 & 4 & 4 & 3 "
        r"\\ "
        "\n "
        r"\addlinespace[0.5em]"
        " \n "
        r"\makecell[l]{Average demonstration\\ duration [s]} "
        f"& ${np.around(np.mean(do_average_dur),3)}"
        r"\pm"
        f"{np.around(np.std(do_average_dur),3)}$ & "
        f"${np.around(np.mean(bpu_average_dur),3)}"
        r"\pm"
        f"{np.around(np.std(bpu_average_dur), 3)}$ &"
        f" ${np.around(np.mean(cr_average_dur), 3)}"
        r"\pm"
        f"{np.around(np.std(cr_average_dur), 3)}$ & "
        f"${np.around(np.mean(db_average_dur), 3)}"
        r"\pm"
        f"{np.around(np.std(db_average_dur), 3)}$ &  "
        f"${np.around(np.mean(dc_average_dur), 3)}"
        r"\pm"
        f"{np.around(np.std(dc_average_dur), 3)}$"
        r"\\ "
        "\n "
        r"\addlinespace[0.5em] "
        "\n "
        r"\makecell[l]{DTW [s]} & "
        f"${np.around(np.mean(do_prep_time), 3)}"
        r"\pm"
        f"{np.around(np.std(do_prep_time), 3)}$ & "
        f"${np.around(np.mean(bpu_prep_time), 3)}"
        r"\pm"
        f"{np.around(np.std(bpu_prep_time), 3)}$ & "
        f"${np.around(np.mean(cr_prep_time), 3)}"
        r"\pm"
        f"{np.around(np.std(cr_prep_time), 3)}$ & "
        f"${np.around(np.mean(db_prep_time), 3)}"
        r"\pm"
        f"{np.around(np.std(db_prep_time), 3)}$ & "
        f" ${np.around(np.mean(dc_prep_time), 3)}"
        r"\pm"
        f"{np.around(np.std(dc_prep_time), 3)}$ "
        r"\\ "
        "\n "
        r"\addlinespace[0.5em] "
        "\n "
        r"\makecell[l]{GMM + GMR [s]} &"
        f" ${np.around(np.mean(do_encod_time), 3)}"
        r"\pm"
        f"{np.around(np.std(do_encod_time), 3)}$ & "
        f"${np.around(np.mean(bpu_encod_time), 3)}"
        r"\pm"
        f"{np.around(np.std(bpu_encod_time), 3)}$ & "
        f"${np.around(np.mean(cr_encod_time), 3)}"
        r"\pm"
        f"{np.around(np.std(cr_encod_time), 3)}$ & "
        f"${np.around(np.mean(db_encod_time), 3)}"
        r"\pm"
        f"{np.around(np.std(db_encod_time), 3)}$ & "
        f" ${np.around(np.mean(dc_encod_time), 3)}"
        r"\pm"
        f"{np.around(np.std(dc_encod_time), 3)}$ "
        r"\\ "
        "\n "
        r"\addlinespace[0.5em] "
        "\n"
        r" \makecell[l]{BO [s]} & "
        f"${np.around(np.mean(do_bo_time), 3)}"
        r"\pm"
        f"{np.around(np.std(do_bo_time), 3)}$ & "
        f"${np.around(np.mean(bpu_bo_time), 3)}"
        r"\pm"
        f"{np.around(np.std(bpu_bo_time), 3)}$ & "
        f"${np.around(np.mean(cr_bo_time), 3)}"
        r"\pm"
        f"{np.around(np.std(cr_bo_time), 3)}$ &"
        f" ${np.around(np.mean(db_bo_time), 3)}"
        r"\pm"
        f"{np.around(np.std(db_bo_time), 3)}$ & "
        f" ${np.around(np.mean(dc_bo_time), 3)}"
        r"\pm"
        f"{np.around(np.std(dc_bo_time), 3)}$ "
        r"\\ "
        "\n"
        r" \addlinespace[0.5em] "
        "\n "
        r" \makecell[l]{ProMP[s]} & "
        f"${np.around(np.mean(do_traj_time), 3)}"
        r"\pm"
        f"{np.around(np.std(do_traj_time), 3)}$ &"
        f" ${np.around(np.mean(bpu_traj_time), 3)}"
        r"\pm"
        f"{np.around(np.std(bpu_traj_time), 3)}$ &"
        f" ${np.around(np.mean(cr_traj_time), 3)}"
        r"\pm"
        f"{np.around(np.std(cr_traj_time), 3)}$ &"
        f" ${np.around(np.mean(db_traj_time), 3)}"
        r"\pm"
        f"{np.around(np.std(db_traj_time), 3)}$ & "
        f" ${np.around(np.mean(dc_traj_time), 3)}"
        r"\pm"
        f"{np.around(np.std(dc_traj_time), 3)}$ "
        r"\\ "
        "\n"
        r" \addlinespace[0.5em] "
        "\n"
        r"\makecell[l]{Total time [s]} "
        f"& ${np.around(np.mean(do_total_time), 3)}"
        r"\pm"
        f"{np.around(np.std(do_total_time), 3)}$ &"
        f" ${np.around(np.mean(bpu_total_time), 3)}"
        r"\pm"
        f"{np.around(np.std(bpu_total_time), 3)}$ &"
        f" ${np.around(np.mean(cr_total_time), 3)}"
        r"\pm"
        f"{np.around(np.std(cr_total_time), 3)}$ &"
        f" ${np.around(np.mean(db_total_time), 3)}"
        r"\pm"
        f"{np.around(np.std(db_total_time), 3)}$  &"
        f"  ${np.around(np.mean(dc_total_time), 3)}"
        r"\pm"
        f"{np.around(np.std(dc_total_time), 3)}$ "
        r"\\ "
        "\n"
        r" \bottomrule "
        "\n "
        r"\end{tabular}"
    )

    with open('door_maintenance_tasks.tex', 'w') as f:
        f.write(text_file)
