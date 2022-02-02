from typing import Union

import RWS2


class RwsWrapper:
    def __init__(self, robot_url: str) -> None:
        """
        CLass constructor.
        :param robot_url: string that defines the robot url
        """
        self.robot = RWS2.RWS(robot_url)

    def set_RAPID_variable(
        self, variable_name: str, new_value: Union[float, int, str]
    ) -> None:
        """
        This method sets a RAPID variable to a new value. Both the variable name and
        the new value are passed by the user as method arguments. The user needs to
        request the controller mastership before changing the variable.
        :param variable_name: name of variable to update/change
        :param new_value: new variable value
        """
        self.robot.request_mastership()
        self.robot.set_rapid_variable(variable_name, new_value)
        self.robot.release_mastership()

    def turn_motors_on(self) -> None:
        """
        This method turns the robot motors on.
        """
        self.robot.request_mastership()
        self.robot.motors_on()
        self.robot.release_mastership()

    def activate_lead_through(self) -> None:
        """
        This method turns the motors on and activate the lead through mode.
        """
        self.turn_motors_on()
        self.robot.activate_lead_through()

    def deactivate__lead_through(self) -> None:
        """
        This method deactivates the lead through mode and switches off the robot motors.
        """
        self.robot.deactivate_lead_through()
        self.robot.motors_off()

    def complete_instruction(
        self, reset_pp: bool = False, var: str = "ready_flag"
    ) -> None:
        """
        This method sets up the robot, starts the RAPID program with a flag specifying
        if the program pointer needs to be reset and then it waits for the task
        completion. Finally, it stops the RAPID program and resumes the settings.
        :param reset_pp: boolean to determine if the program pointer needs to be reset
        :param var: RAPID variable that helps to synchronize the python script and the
                    RAPID program to achieve a coherent task execution
        """
        self.turn_motors_on()
        self.robot.start_RAPID(reset_pp)
        self.robot.wait_for_rapid()
        self.robot.stop_RAPID()
        self.set_RAPID_variable(var, "FALSE")
