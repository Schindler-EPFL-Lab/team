import argparse
import os.path

from learning_from_demo.demonstration_player import DemonstrationPlayer
from learning_from_demo.trajectory import Trajectory

# Create object for parsing command-line options
parser = argparse.ArgumentParser(
    description="Playback a saved robot trajectory."
)

# Add argument which takes the url to be used for the communication
parser.add_argument("--url", type=str, required=True, help="Url address to send "
                                                           "requests to")
# Add argument which takes the file destination path to be used for data storing
parser.add_argument("--demo_path", type=str, required=True, help="Demonstration path")

# Parse the command line arguments to an object
args = parser.parse_args()

# Safety if some parameters have not been given
if not args.url or not args.demo_path:
    print("No all the necessary parameters have been given.")
    print("For help type --help")
    exit()

# Check if the given file has json extension
if os.path.splitext(args.demo_path)[1] != ".json":
    print("The given file is not of correct file format.")
    print("Only .json files are accepted")
    exit()

try:
    play = DemonstrationPlayer(base_url=args.url)
    trajectory = Trajectory.from_file(args.demo_path)
    play.play(trajectory)
finally:
    pass
