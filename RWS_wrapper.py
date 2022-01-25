import RWS2
import string


class RwsWrapper:
    def __init__(self, robot_url):
        self.robot = RWS2.RWS(robot_url)

    def set_RAPID_variable(self, variable_name: string, new_value):
        self.robot.request_mastership()
        self.robot.set_rapid_variable(variable_name, new_value)
        self.robot.release_mastership()

    def turn_motors_on(self):
        self.robot.request_mastership()
        self.robot.motors_on()
        self.robot.release_mastership()

    def complete_instruction(self, var="ready_flag"):
        self.turn_motors_on()
        self.robot.start_RAPID()
        self.robot.wait_for_rapid()
        self.robot.stop_RAPID()
        self.set_RAPID_variable(var, "FALSE")
