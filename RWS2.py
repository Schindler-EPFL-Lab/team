import ast
import json
import math
from typing import Union

import xmltodict
from requests.auth import HTTPBasicAuth
from requests import Session, Response

from arco.utility.logger import log


class RWS:
    """Class for communicating with RobotWare through Robot Web Services
    (ABB's Rest API)
    """

    def __init__(
        self, base_url: str, username: str = "Default User", password: str = "robotics"
    ) -> None:
        """
        Class constructor.
        :param base_url: base url to address the requests
        :param username: authentication username
        :param password: authentication password
        """
        self.base_url = base_url
        self.username = username
        self.password = password
        self.session = Session()  # create persistent HTTP communication
        self.session.auth = HTTPBasicAuth(self.username, self.password)
        self.session.headers = {
            "Accept": "application/xhtml+xml;v=2.0",
            "Content-Type": "application/x-www-form-urlencoded;v=2.0",
        }
        self.session.verify = False

    def set_rapid_variable(self, var: str, value: Union[str, float, int]) -> Response:
        """Sets the value of any RAPID variable.
        Unless the variable is of type 'num', 'value' has to be a string.
        """

        payload = {"value": value}
        resp = self.session.post(
            self.base_url + "/rw/rapid/symbol/RAPID/T_ROB1/" + var + "/data",
            data=payload,
        )
        return resp

    def get_rapid_variable(self, var: str) -> Union[str, float]:
        """Gets the raw value of any RAPID variable.
        """

        resp = self.session.get(
            self.base_url + "/rw/rapid/symbol/RAPID/T_ROB1/" + var + "/data?value=1"
        )
        _dict = xmltodict.parse(resp.content)
        value = _dict["html"]["body"]["div"]["ul"]["li"]["span"]["#text"]
        return value

    def get_robtarget_variables(self, var: str) -> (list[float], list[float]):
        """Gets both translational and rotational data from robtarget.
        """

        resp = self.session.get(
            self.base_url + "/rw/rapid/symbol/RAPID/T_ROB1/" + var + "/data?value=1"
        )
        _dict = xmltodict.parse(resp.content)
        data = _dict["html"]["body"]["div"]["ul"]["li"]["span"]["#text"]
        data_list = ast.literal_eval(data)  # Convert the pure string from data to list
        trans = data_list[0]  # Get x,y,z from robtarget relative to work object (table)
        rot = data_list[1]  # Get orientation of robtarget
        return trans, rot

    def get_gripper_position(self) -> (list[float], list[float]):
        """Gets translational and rotational of the UiS tool 'tGripper'
        with respect to the work object 'wobjTableN'.
        """

        resp = self.session.get(
            self.base_url + "/rw/motionsystem/mechunits/ROB_1/robtarget/"
            "?tool=tGripper&wobj=wobjTableN&coordinate=Wobj&json=1"
        )
        json_string = resp.text
        _dict = json.loads(json_string)
        data = _dict["_embedded"]["_state"][0]
        trans = [data["x"], data["y"], data["z"]]
        trans = [float(i) for i in trans]
        rot = [data["q1"], data["q2"], data["q3"], data["q4"]]
        rot = [float(i) for i in rot]

        return trans, rot

    def get_gripper_height(self) -> float:
        """Extracts only the height from gripper position.
        (See get_gripper_position)
        """

        trans, rot = self.get_gripper_position()
        height = trans[2]

        return height

    def set_robtarget_translation(
        self, var: str, trans: Union[list[float], tuple[float]]
    ) -> None:
        """Sets the translational data of a robtarget variable in RAPID.
        """

        _trans, rot = self.get_robtarget_variables(var)
        if rot == [0, 0, 0, 0]:  # If the target has no previously defined orientation
            self.set_rapid_variable(
                var,
                "[["
                + ",".join([str(s) for s in trans])
                + "],[0,1,0,0],[-1,0,0,0],[9E+9,9E+9,9E+9,9E+9,9E+9,9E+9]]",
            )
        else:
            self.set_rapid_variable(
                var,
                "[["
                + ",".join([str(s) for s in trans])
                + "],["
                + ",".join(str(s) for s in rot)
                + "],[-1,0,0,0],[9E+9,9E+9,9E+9,9E+9,"
                "9E+9,9E+9]]",
            )

    def set_robtarget_rotation_z_degrees(
        self, var: str, rotation_z_degrees: float
    ) -> None:
        """Updates the orientation of a robtarget variable
        in RAPID by rotation about the z-axis in degrees.
        """

        rot = z_degrees_to_quaternion(rotation_z_degrees)

        trans, _rot = self.get_robtarget_variables(var)

        self.set_rapid_variable(
            var,
            "[["
            + ",".join([str(s) for s in trans])
            + "],["
            + ",".join(str(s) for s in rot)
            + "],[-1,0,0,0],[9E+9,9E+9,9E+9,9E+9,"
            "9E+9,9E+9]]",
        )

    def set_robtarget_rotation_quaternion(
        self, var: str, rotation_quaternion: Union[list[float], tuple[float]]
    ) -> None:
        """Updates the orientation of a robtarget variable in RAPID by a Quaternion.
        """

        trans, _rot = self.get_robtarget_variables(var)

        self.set_rapid_variable(
            var,
            "[["
            + ",".join([str(s) for s in trans])
            + "],["
            + ",".join(str(s) for s in rotation_quaternion)
            + "],[-1,0,0,0],[9E+9,"
            "9E+9,9E+9,9E+9,9E+9,"
            "9E+9]]",
        )

    def wait_for_rapid(self, var: str = "ready_flag") -> None:
        """Waits for robot to complete RAPID instructions
        until boolean variable in RAPID is set to 'TRUE'.
        Default variable name is 'ready_flag', but others may be used.
        """

        while self.get_rapid_variable(var) == "FALSE" and self.is_running():
            # read robot data-> here just printing to the console
            self.log_robot_data()

    def set_rapid_array(
        self, var: str, value: Union[list[float], tuple[float]]
    ) -> None:
        """Sets the values of a RAPID array by sending a list from Python.
        """

        self.set_rapid_variable(var, "[" + ",".join([str(s) for s in value]) + "]")

    def reset_pp(self) -> None:
        """Resets the program pointer to main procedure in RAPID.
        """

        resp = self.session.post(
            self.base_url + "/rw/rapid/execution/resetpp?mastership=implicit"
        )
        if resp.status_code == 204:
            print("Program pointer reset to main")
        else:
            print("Could not reset program pointer to main")

    def request_mastership(self) -> None:
        self.session.post(self.base_url + "/rw/mastership/request")

    def release_mastership(self) -> None:
        self.session.post(self.base_url + "/rw/mastership/release",)

    def request_rmmp(self) -> None:
        self.session.post(self.base_url + "/users/rmmp", data={"privilege": "modify"})

    def cancel_rmmp(self) -> None:
        self.session.post(self.base_url + "/users/rmmp/cancel")

    def motors_on(self) -> None:
        """Turns the robot's motors on.
        Operation mode has to be AUTO.
        """

        payload = {"ctrl-state": "motoron"}
        resp = self.session.post(
            self.base_url + "/rw/panel/ctrl-state?ctrl-state=motoron", data=payload,
        )

        if resp.status_code == 204:
            print("Robot motors turned on")
        else:
            print("Could not turn on motors. The controller might be in manual mode")

    def motors_off(self) -> None:
        """Turns the robot's motors off.
        """

        payload = {"ctrl-state": "motoroff"}
        resp = self.session.post(
            self.base_url + "/rw/panel/ctrl-state?ctrl-state=motoroff", data=payload
        )

        if resp.status_code == 204:
            print("Robot motors turned off")
        else:
            print("Could not turn off motors")

    def start_RAPID(self, pp_to_reset: bool) -> None:
        """Resets program pointer to main procedure in RAPID and starts RAPID execution.
        """
        if pp_to_reset:
            self.reset_pp()
        payload = {
            "regain": "continue",
            "execmode": "continue",
            "cycle": "once",
            "condition": "none",
            "stopatbp": "disabled",
            "alltaskbytsp": "false",
        }
        resp = self.session.post(
            self.base_url + "/rw/rapid/execution/start?mastership=implicit",
            data=payload,
        )
        if resp.status_code == 204:
            print("RAPID execution started from main")
        else:
            opmode = self.get_operation_mode()
            ctrlstate = self.get_controller_state()

            print(
                f"""
            Could not start RAPID. Possible causes:
            * Operating mode might not be AUTO. Current opmode: {opmode}.
            * Motors might be turned off. Current ctrlstate: {ctrlstate}.
            * RAPID might have write access.
            """
            )

    def stop_RAPID(self) -> None:
        """Stops RAPID execution.
        """

        payload = {"stopmode": "stop", "usetsp": "normal"}
        resp = self.session.post(
            self.base_url + "/rw/rapid/execution/stop", data=payload,
        )
        if resp.status_code == 204:
            print("RAPID execution stopped")
        else:
            print("Could not stop RAPID execution")

    def get_execution_state(self) -> str:
        """Gets the execution state of the controller.
        """

        resp = self.session.get(self.base_url + "/rw/rapid/execution?json=1",)
        _dict = xmltodict.parse(resp.content)
        data = _dict["html"]["body"]["div"]["ul"]["li"]["span"][0]["#text"]
        return data

    def is_running(self) -> bool:
        """Checks and returns the execution state of the controller.
        """

        execution_state = self.get_execution_state()
        if execution_state == "running":
            return True
        else:
            return False

    def get_operation_mode(self) -> str:
        """Gets the operation mode of the controller.
        """

        resp = self.session.get(self.base_url + "/rw/panel/opmode")
        _dict = xmltodict.parse(resp.content)
        data = _dict["html"]["body"]["div"]["ul"]["li"][0]["span"]["#text"]
        return data

    def get_controller_state(self) -> str:
        """Gets the controller state.
        """

        resp = self.session.get(self.base_url + "/rw/panel/ctrl-state")
        _dict = xmltodict.parse(resp.content)
        data = _dict["html"]["body"]["div"]["ul"]["li"]["span"]["#text"]
        return data

    def set_speed_ratio(self, speed_ratio: float) -> None:
        """Sets the speed ratio of the controller.
        """

        if not 0 < speed_ratio <= 100:
            print("You have entered a false speed ratio value! Try again.")
            return

        payload = {"speed-ratio": speed_ratio}
        resp = self.session.post(
            self.base_url + "/rw/panel/speedratio?mastership=implicit", data=payload
        )
        if resp.status_code == 204:
            print(f"Set speed ratio to {speed_ratio}%")
        else:
            print("Could not set speed ratio!")

    def set_zonedata(self, var: str, zonedata: Union[float, str]) -> None:
        """Sets the zonedata of a zonedata variable in RAPID.
        """

        if zonedata not in ["fine", 0, 1, 5, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200]:
            print("You have entered false zonedata! Please try again")
            return
        else:
            if zonedata in [10, 20, 30, 40, 50, 60, 80, 100, 150, 200]:
                value = (
                    f"[FALSE, {zonedata}, {zonedata * 1.5}, {zonedata * 1.5}, "
                    f"{zonedata * 0.15}, {zonedata * 1.5}, {zonedata * 0.15}]"
                )
            elif zonedata == 0:
                value = (
                    f"[FALSE, {zonedata + 0.3}, {zonedata + 0.3}, {zonedata + 0.3}, "
                    f"{zonedata + 0.03}, {zonedata + 0.3}, {zonedata + 0.03}]"
                )
            elif zonedata == 1:
                value = (
                    f"[FALSE, {zonedata}, {zonedata}, {zonedata}, {zonedata * 0.1},"
                    f" {zonedata}, {zonedata * 0.1}]"
                )
            elif zonedata == 5:
                value = (
                    f"[FALSE, {zonedata}, {zonedata * 1.6}, {zonedata * 1.6}, "
                    f"{zonedata * 0.16}, {zonedata * 1.6}, {zonedata * 0.16}]"
                )
            else:  # zonedata == 'fine':
                value = f"[TRUE, {0}, {0}, {0}, {0}, {0}, {0}]"

        resp = self.set_rapid_variable(var, value)
        if resp.status_code == 204:
            print(f'Set "{var}" zonedata to z{zonedata}')
        else:
            print("Could not set zonedata! Check that the variable name is correct")

    def set_speeddata(self, var: str, speeddata: float) -> None:
        """Sets the speeddata of a speeddata variable in RAPID.
        """

        resp = self.set_rapid_variable(var, f"[{speeddata},500,5000,1000]")
        if resp.status_code == 204:
            print(f'Set "{var}" speeddata to v{speeddata}')
        else:
            print("Could not set speeddata. Check that the variable name is correct")

    def log_robot_data(self) -> None:
        """Retrieve robot data and print it to console.
        """
        tcp_pos, tcp_ori = self.get_tcp_pose()
        joints_pos = self.get_joints_positions()
        log.info(
            f"\n"
            f"robot tcp position {tcp_pos} \n"
            f"robot tcp orientation {tcp_ori} \n"
            f"robot joints position {joints_pos}"
        )

    def get_joints_positions(
        self, n_joints: int = 6, mechunits: str = "ROB_1"
    ) -> list[float]:
        """Gets the robot joints positions in degrees.
        """

        resp = self.session.get(
            self.base_url
            + "/rw/motionsystem/mechunits/"
            + mechunits
            + "/jointtarget?ignore=1"
        )
        _dict = xmltodict.parse(resp.content)
        joints_pos = []
        for i in range(n_joints):
            joints_pos.append(
                _dict["html"]["body"]["div"]["ul"]["li"]["span"][i]["#text"]
            )
        return joints_pos

    def get_tcp_pose(
        self,
        mechunits: str = "ROB_1",
        tool: str = "tool0",
        wobj: str = "wobj0",
        frame: str = "Base",
    ) -> (list[float], list[float]):
        """Gets the robot tcp position (mm) and orientation (quaternions).
        """

        resp = self.session.get(
            self.base_url
            + "/rw/motionsystem/mechunits/"
            + mechunits
            + "/robtarget?tool="
            + tool
            + "&wobj="
            + wobj
            + "&coordinate="
            + frame
        )
        _dict = xmltodict.parse(resp.content)
        tcp_pos = []
        tcp_ori = []
        # (x,y,z) are stored as the first three values in the xml file
        for i in range(3):
            tcp_pos.append(_dict["html"]["body"]["div"]["ul"]["li"]["span"][i]["#text"])
        # (q1,q2,q3, q4) are stored as the 4/5/6/7 values in the xml file
        for j in range(3, 7):
            tcp_ori.append(_dict["html"]["body"]["div"]["ul"]["li"]["span"][j]["#text"])
        return tcp_pos, tcp_ori


def z_degrees_to_quaternion(rotation_z_degrees: float) -> list[float]:
    """Convert a rotation about the z-axis in degrees to Quaternion.
    """
    roll = math.pi
    pitch = 0
    yaw = math.radians(rotation_z_degrees)

    qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + math.sin(
        roll / 2
    ) * math.sin(pitch / 2) * math.sin(yaw / 2)
    qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - math.cos(
        roll / 2
    ) * math.sin(pitch / 2) * math.sin(yaw / 2)
    qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + math.sin(
        roll / 2
    ) * math.cos(pitch / 2) * math.sin(yaw / 2)
    qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - math.sin(
        roll / 2
    ) * math.sin(pitch / 2) * math.cos(yaw / 2)

    return [qw, qx, qy, qz]
