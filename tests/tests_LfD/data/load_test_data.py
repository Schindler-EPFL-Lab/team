import subprocess


def main():
    subprocess.run(
        [
            "dvc",
            "get",
            "git@ssh.dev.azure.com:v3/devsdb/CRD-NT_ARCO/arco-datasets",
            "arco_control/learning_from_demonstrations/approaching/"
            "approaching_amk_1.json",
            "-o",
            "tests/tests_LfD/data/approaching_amk_1.json",
        ]
    )

    subprocess.run(
        [
            "dvc",
            "get",
            "git@ssh.dev.azure.com:v3/devsdb/CRD-NT_ARCO/arco-datasets",
            "arco_control/learning_from_demonstrations/approaching/"
            "approaching_mm_1.json",
            "-o",
            "tests/tests_LfD/data/approaching_mm_1.json",
        ]
    )

    subprocess.run(
        [
            "dvc",
            "get",
            "git@ssh.dev.azure.com:v3/devsdb/CRD-NT_ARCO/arco-datasets",
            "arco_control/learning_from_demonstrations/approaching/"
            "approaching_lp_1.json",
            "-o",
            "tests/tests_LfD/data/approaching_lp_1.json",
        ]
    )


if __name__ == "__main__":
    main()
