from arco.utility.handling_data import get_demo_files, check_data_timestamps, \
    check_nan_values, check_reading_files


def main() -> None:
    """
    Runs all the validation checks for each single file in the recorded demonstration
    dataset.
    """
    list_files = get_demo_files()
    # loop over each file path
    for file in list_files:
        check_reading_files(file)
        check_nan_values(file)
        check_data_timestamps(file)


if __name__ == "__main__":
    main()
