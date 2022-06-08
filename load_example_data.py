import subprocess


def main():
    subprocess.run(
        [
            "dvc",
            "get",
            "git@ssh.dev.azure.com:v3/devsdb/CRD-NT_ARCO/arco-datasets",
            "arco_control/learning_from_demonstrations/door_opening"
            "/door_opening_lp_1.json",
            "-o",
            "demonstrations/door_opening/door_opening_lp_1.json",
        ]
    )

    subprocess.run(
        [
            "dvc",
            "get",
            "git@ssh.dev.azure.com:v3/devsdb/CRD-NT_ARCO/arco-datasets",
            "arco_control/learning_from_demonstrations/door_opening"
            "/door_opening_lp_2.json",
            "-o",
            "demonstrations/door_opening/door_opening_lp_2.json",
        ]
    )

    subprocess.run(
        [
            "dvc",
            "get",
            "git@ssh.dev.azure.com:v3/devsdb/CRD-NT_ARCO/arco-datasets",
            "arco_control/learning_from_demonstrations/door_opening"
            "/door_opening_lp_3.json",
            "-o",
            "demonstrations/door_opening/door_opening_lp_3.json",
        ]
    )


if __name__ == "__main__":
    main()
