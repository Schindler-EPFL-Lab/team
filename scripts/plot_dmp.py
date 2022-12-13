import argparse
import pathlib

from team.dynamical_movement_primitives import DynamicMovementPrimitives
from team.utility.dmp_visualization import plotting


def get_path() -> tuple[str, bool, str]:
    parser = argparse.ArgumentParser(description="Plot DMP")
    parser.add_argument("--path", type=str, required=True, help="Path to saved DMP")
    parser.add_argument("--show", type=bool, default=True, help="Show the plot")
    parser.add_argument(
        "--save_to", type=str, default="data/saved", help="Path to save image to"
    )
    args = parser.parse_args()

    return args.path, args.show, args.save_to


def main():
    path_to_dmp, show, save_to = get_path()
    dmp = DynamicMovementPrimitives.load_dmp(pathlib.Path(path_to_dmp))
    plotting(dmp, show, save_to)


if __name__ == "__main__":
    main()
