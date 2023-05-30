import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index", type=int, default=1, help="Index of reproduction trajectory"
    )
    parser.add_argument("--task", type=str, default="rail_cleaning", help="Task name")
    args = parser.parse_args()
    i = args.index
    if not 1 <= i <= 30:
        raise ValueError("index should be in range [1-30]")
    task = args.task
    if task not in ["rail_cleaning", "brush_homing", "brush_picking_up", "opening"]:
        raise ValueError(
            "task name should be in [rail_cleaning,"
            "brush_homing, brush_picking_up, opening]"
        )
    base_folder = Path(__file__).parent.parent.parent
    data_folder = base_folder.joinpath("data/maintenance_tasks/maintenance_noise_20.0")
    task_folder = data_folder.joinpath(task)
    regression_path = task_folder.joinpath("dmp_data.json")
    repro_path = task_folder.joinpath(f"reproduction{i}/task_config.json")
    # load regression
    with open(regression_path, "r") as f:
        dmp_data = json.load(f)
        regression = np.array(dmp_data["regression"])
    # load reproduction
    with open(repro_path, "r") as f:
        repro_data = json.load(f)
        reproduction = np.array(repro_data["trajectory"])
    # plot data
    fig, axs = plt.subplots(2, 3, layout="constrained", figsize=(18, 12))
    x = 0
    y = 0
    for i in range(np.shape(regression)[1] - 1):
        axs[x, y].plot(
            regression[:, 0], regression[:, i + 1], linewidth=5, label="regression"
        )
        axs[x, y].plot(
            regression[:, 0], reproduction[:, i], linewidth=5, label="reproduction"
        )
        axs[x, y].tick_params(axis="x", labelsize=14)
        axs[x, y].tick_params(axis="y", labelsize=14)
        axs[x, y].set_title(f"Joint {i + 1} evolution", fontsize=16)
        axs[x, y].legend(fontsize=14, markerscale=3)
        y = y + 1
        if y == 3:
            x = 1
            y = 0
    fig.supxlabel("Time [s]", fontsize=16)
    fig.supylabel("Joint angle [deg]", fontsize=16)
    plt.savefig("noise_on_joints.pgf", backend="pgf")
    plt.show()


if __name__ == "__main__":
    main()
