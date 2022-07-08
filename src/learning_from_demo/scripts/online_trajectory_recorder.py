import argparse
import os.path
from timeit import default_timer as timer

from learning_from_demo.recorders.demonstration_recorder import DemonstrationRecorder
from learning_from_demo.utility.handling_data import create_default_dict
from rws2.RWS2 import RWS

# Create object for parsing command-line options
parser = argparse.ArgumentParser(
    description="Record a robot trajectory demonstrated by the user.\
                 Robot end effector, joints and configurations \
                 information are stored to a file."
)

# Add argument which takes the url to be used for the communication
parser.add_argument(
    "--url", type=str, required=True, help="Url address to send " "requests to"
)
# Add argument which takes the file destination path to be used for data storing
parser.add_argument(
    "--dest_path", type=str, required=True, help="Destination path to" "save data"
)
# Parse the command line arguments to an object
args = parser.parse_args()

# Safety if some parameters have not been given
if not args.url or not args.dest_path:
    print("No all the necessary parameters have been given.")
    print("For help type --help")
    exit()

# Check if the given file has json extension
if os.path.splitext(args.dest_path)[1] != ".json":
    print("The given file is not of correct file format.")
    print("Only .json files are accepted")
    exit()


def main():
    try:

        record = DemonstrationRecorder(data_path=args.dest_path)

        rws_robot = RWS(args.url)
        data = create_default_dict()
        # wait for the RAPID program to start
        while not rws_robot.is_running():
            pass
        # since RWS2 doesn't return a timestamp with the measurement
        # compute it from Python
        t_start = timer()
        # record until RAPID program stops
        while rws_robot.is_running():
            tcp_pos, tcp_ori, rob_cf = rws_robot.get_tcp_info()
            joints = rws_robot.get_joints_positions()
            timestamp = [timer() - t_start]
            values_list = timestamp + tcp_pos + tcp_ori + rob_cf + joints
            # check that all information are available
            if tcp_pos and tcp_ori and rob_cf and joints:
                info = {key: value for (key, value) in zip(data.keys(), values_list)}
                record.update(tmp_dict=info)
        # save data to file
        record.create_file()

    finally:
        pass


if __name__ == "__main__":
    main()
