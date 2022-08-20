import argparse
from timeit import default_timer as timer

import keyboard
from rws2.RWS_wrapper import RwsWrapper

from team.recorders.demonstration_recorder import DemonstrationRecorder
from team.utility.handling_data import create_default_dict

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

try:
    record = DemonstrationRecorder(data_path=args.dest_path)

    rws = RwsWrapper(args.url)
    data = create_default_dict()
    var = "ready_flag"
    # setup robot mode
    rws.activate_lead_through()
    # since RWS2 doesn't return a timestamp with the measurement, compute it from Python
    t_start = timer()
    while True:
        tcp_pos, tcp_ori, rob_cf = rws.robot.get_tcp_info()
        joints = rws.robot.get_joints_positions()
        timestamp = [timer() - t_start]
        values_list = timestamp + tcp_pos + tcp_ori + rob_cf + joints
        # check that all information are available
        if tcp_pos and tcp_ori and rob_cf and joints:
            info = {key: value for (key, value) in zip(data.keys(), values_list)}
            record.update(tmp_dict=info)
        if keyboard.is_pressed("q"):
            break
    # robot shutdown operations
    rws.deactivate_lead_through()
    rws.set_RAPID_variable(var, "FALSE")
    # save data to file
    record.create_file()

finally:
    pass
