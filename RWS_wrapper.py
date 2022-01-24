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


if __name__ == "__main__":
    base_url = "https://localhost:8881"  # https://192.168.125.1 real robot
    home = (
        "[[600, 0.00, 800], [-0.500, 0.0, -0.866, 0.0], [0, 0, 0, 0], "
        "[9E+9,9E+9,9E+9,9E+9,9E+9,9E+9]]"
    )
    Loc1 = (
        "[[600, 200.00, 800], [-0.500, 0.0, -0.866, 0.0], [0, 0, 0, 0], "
        "[9E+9,9E+9,9E+9,9E+9,9E+9,9E+9]]"
    )
    Loc2 = (
        "[[500, 200.00, 800], [-0.500, 0.0, -0.866, 0.0], [0, 0, 0, 0], "
        "[9E+9,9E+9,9E+9,9E+9,9E+9,9E+9]]"
    )
    Loc3 = (
        "[[500, -200.00, 800], [-0.500, 0.0, -0.866, 0.0], [0, 0, 0, 0], "
        "[9E+9,9E+9,9E+9,9E+9,9E+9,9E+9]]"
    )
    Loc4 = (
        "[[600, -200.00, 800], [-0.500, 0.0, -0.866, 0.0], [0, 0, 0, 0],"
        "[9E+9,9E+9,9E+9,9E+9,9E+9,9E+9]]"
    )
    rws = RwsWrapper(base_url)  # verify certificate set to False

    # TEST RAPID PROGRAM
    rws.set_RAPID_variable("program_running", "TRUE")
    rws.set_RAPID_variable("Loc", home)
    rws.complete_instruction()
    rws.set_RAPID_variable("Loc", Loc1)
    rws.complete_instruction()
    rws.set_RAPID_variable("Loc", Loc2)
    rws.complete_instruction()
    rws.set_RAPID_variable("Loc", Loc3)
    rws.complete_instruction()
    rws.set_RAPID_variable("Loc", Loc4)
    rws.complete_instruction()
    rws.set_RAPID_variable("Loc", home)
    rws.complete_instruction()
    rws.set_RAPID_variable("program_running", "FALSE")
    rws.complete_instruction()
    rws.robot.motors_off()
