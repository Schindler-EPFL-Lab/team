from RWS_wrapper import RwsWrapper

if __name__ == "__main__":
    virtual_controller_url = "https://localhost:8881"
    robot_url = "https://192.168.125.1"
    home = (
        "[[600, 0.00, 800], [-0.500, 0.0, -0.866, 0.0], [0, 0, 0, 0], "
        "[9E+9,9E+9,9E+9,9E+9,9E+9,9E+9]]"
    )
    loc1 = (
        "[[600, 200.00, 800], [-0.500, 0.0, -0.866, 0.0], [0, 0, 0, 0], "
        "[9E+9,9E+9,9E+9,9E+9,9E+9,9E+9]]"
    )
    loc2 = (
        "[[500, 200.00, 800], [-0.500, 0.0, -0.866, 0.0], [0, 0, 0, 0], "
        "[9E+9,9E+9,9E+9,9E+9,9E+9,9E+9]]"
    )
    loc3 = (
        "[[500, -200.00, 800], [-0.500, 0.0, -0.866, 0.0], [0, 0, 0, 0], "
        "[9E+9,9E+9,9E+9,9E+9,9E+9,9E+9]]"
    )
    loc4 = (
        "[[600, -200.00, 800], [-0.500, 0.0, -0.866, 0.0], [0, 0, 0, 0],"
        "[9E+9,9E+9,9E+9,9E+9,9E+9,9E+9]]"
    )
    rws = RwsWrapper(virtual_controller_url)  # verify certificate set to False

    # TEST RAPID PROGRAM ->the robot spans the sides of a rectangle in the xy plane
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
