import argparse
import json
import statistics
from pathlib import Path 

from compare import get_gmcc_score_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction)
    parser.add_argument("--filename", type=str, default="task_config.json")
    parser.add_argument("--datafolder", type=str, default="Ebikon")
    args = parser.parse_args()

    root_folder = Path(__file__).parent.parent.parent.resolve()
    data_dir = Path(root_folder, "data", args.datafolder)
    folders = [f for f in data_dir.iterdir() if f.is_dir()]
    for task in folders:
        gmccs = []
        for task_execution in [f for f in task.iterdir() if f.is_dir()]:
            gmcc, std_dev = get_gmcc_score_folder(
                data_path=task_execution,
                rewrite=args.rewrite,
                filename=args.filename,
            )
            gmccs.append(gmcc)
        with open(Path(task, "gmccs.json"), "w") as f:
            json.dump(
                {"gmccs": statistics.mean(gmccs), "std": statistics.stdev(gmccs)}, f
            )

