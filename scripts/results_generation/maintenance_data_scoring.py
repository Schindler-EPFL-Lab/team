import argparse
import json
import statistics
from pathlib import Path

from compare import get_score_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewrite-gmcc", action=argparse.BooleanOptionalAction)
    parser.add_argument("--filename", type=str, default="task_config.json")
    parser.add_argument("--datafolder", type=str, default="maintenance_tasks")
    args = parser.parse_args()

    root_folder = Path(__file__).parent.parent.parent.resolve()
    data_dir = Path(root_folder, "data", args.datafolder)
    folders = [f for f in data_dir.iterdir() if f.is_dir()]
    for task in folders:
        gmccs = []
        joint_errors = []
        for task_execution in [f for f in task.iterdir() if f.is_dir()]:
            print(task_execution)
            gmcc, std_dev, joint_error, joint_std = get_score_folder(
                data_path=task_execution,
                filename=args.filename,
                rewrite_gmcc=args.rewrite_gmcc,
            )
            gmccs.append(gmcc)
            joint_errors.append(joint_error)
        if Path(task, "gmccs.json").exists():
            Path(task, "gmccs.json").rename("scores.json")
        with open(Path(task, "scores.json"), "w") as f:
            json.dump(
                {
                    "gmccs": {
                        "mean": statistics.mean(gmccs),
                        "std": statistics.stdev(gmccs),
                    },
                    "joints mean absolute error": {
                        "mean": statistics.mean(joint_errors),
                        "std": statistics.stdev(joint_errors),
                    },
                },
                f,
            )
