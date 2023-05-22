import argparse
from pathlib import Path

from compare import get_gmcc_score_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction)
    parser.add_argument("--filename", type=str, default="task_config.json")
    parser.add_argument("--datafolder", type=str, default="maintenance_tasks")
    args = parser.parse_args()

    root_folder = Path(__file__).parent.parent.parent.resolve()
    data_dir = Path(root_folder, "data", args.datafolder)
    folders = [f for f in data_dir.iterdir() if f.is_dir()]
    for task in folders:
        gmcc, std_dev = get_gmcc_score_folder(
            data_path=task, rewrite=args.rewrite, filename=args.filename
        )
