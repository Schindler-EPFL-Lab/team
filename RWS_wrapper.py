from typing import Union

import RWS2


class RwsWrapper:
    def __init__(self, robot_url: str):
        self.robot = RWS2.RWS(robot_url)

    def set_RAPID_variable(
        self, variable_name: str, new_value: Union[float, int, str]
    ) -> None:
        self.robot.request_mastership()
        self.robot.set_rapid_variable(variable_name, new_value)
        self.robot.release_mastership()

    def turn_motors_on(self) -> None:
        self.robot.request_mastership()
        self.robot.motors_on()
        self.robot.release_mastership()

    def complete_instruction(
        self, reset_pp: bool = False, var: str = "ready_flag"
    ) -> None:
        self.turn_motors_on()
        self.robot.start_RAPID(reset_pp)
        self.robot.wait_for_rapid()
        self.robot.stop_RAPID()
        self.set_RAPID_variable(var, "FALSE")
