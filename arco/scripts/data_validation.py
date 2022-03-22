from arco.utility.handling_data import get_demo_files, check_data_timestamps, \
    check_nan_values, check_reading_files


if __name__ == "__main__":
    list_files = get_demo_files()
    check_reading_files(list_files)
    check_nan_values(list_files)
    check_data_timestamps(list_files)
