from RWS_wrapper import RwsWrapper

if __name__ == "__main__":
    url = "https://localhost:8881"  # virtual controller url
    # url = "https://192.168.125.1"  # real robot url
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
    rws = RwsWrapper(url)

    # TEST RAPID PROGRAM ->the robot spans the sides of a rectangle in the xy plane
    path_sequence = [
        (home, True),
        (loc1, False),
        (loc2, False),
        (loc3, False),
        (loc4, False),
        (home, False),
    ]
    rws.set_RAPID_variable("program_running", "TRUE")
    for waypoint, reset_program in path_sequence:
        rws.set_RAPID_variable("Loc", waypoint)
        rws.complete_instruction(reset_program)
    rws.robot.motors_off()
