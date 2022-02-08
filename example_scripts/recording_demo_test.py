import time

import keyboard

from rws2.RWS_wrapper import RwsWrapper
from arco.learning_from_demo.demonstration_recorder import DemonstrationRecorder
from arco.utility.handling_data import create_default_dict

path_to_store = "sample_test.json"
url = "https://localhost:8881"


if __name__ == "__main__":
    record = DemonstrationRecorder(path_to_store_demo=path_to_store)
    rws = RwsWrapper(url)
    data = create_default_dict()
    var = "ready_flag"
    rws.activate_lead_through()
    t_start = time.time()
    while True:
        tcp_pos, tcp_ori, rob_cf = rws.robot.get_tcp_info()
        joints = rws.robot.get_joints_positions()
        timestamp = [time.time() - t_start]
        values_list = timestamp + tcp_pos + tcp_ori + rob_cf + joints
        # check that all information are available
        if tcp_pos and tcp_ori and rob_cf and joints:
            info = {key: value for (key, value) in zip(data.keys(), values_list)}
            record.update(tmp_dict=info)
        if keyboard.is_pressed("q"):
            break
    # robot shutdown operations
    rws.deactivate__lead_through()
    rws.set_RAPID_variable(var, "FALSE")
    # save data to file
    record.create_file()
