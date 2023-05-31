import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    data = np.vstack([regression.T, reproduction.T])

    df = pd.DataFrame(
        data.T,
        columns=[
            "time",
            "reg1",
            "reg2",
            "reg3",
            "reg4",
            "reg5",
            "reg6",
            "rep1",
            "rep2",
            "rep3",
            "rep4",
            "rep5",
            "rep6",
        ],
    )

    tfile = open("trajectory.dat", "w")
    tfile.write(df.to_string())
    tfile.close()


if __name__ == "__main__":
    main()
