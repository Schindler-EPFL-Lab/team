import unittest

import json
import numpy as np
import pandas as pd

from arco.utility.handling_data import get_demo_files, create_default_dict


# unittest will test all the methods whose name starts with 'test'
class DataTest(unittest.TestCase):

    # return True or False
    def test_data_timestamps(self):
        list_file_paths = get_demo_files()
        # analyse each single file
        for data_path in list_file_paths:
            with open(data_path, 'r') as f:
                # load data and extract time vector
                data = json.load(f)
                time = np.array(data["timestamp"])
                time -= time[0]
                delta_t = time[1:] - time[0:-1]
                # check that time is updated at every data reading step
                self.assertEqual(len(np.argwhere(delta_t == 0)), 0)

    def test_nan_values(self):
        # consider empty string and numpy.inf as na values
        pd.set_option('mode.use_inf_as_na', True)
        list_file_paths = get_demo_files()
        # analyse each single file
        for data_path in list_file_paths:
            # load data as pandas dataframe
            df = pd.read_json(data_path)
            self.assertFalse(df.isnull().values.any())

    def test_reading_files(self):
        list_file_paths = get_demo_files()
        # analyse each single file
        for data_path in list_file_paths:
            # load data as pandas dataframe
            df = pd.read_json(data_path)
            default_dict = create_default_dict()
            # checks that the dataframe has the same keys as the recorded dictionary
            for df_key, dict_key in zip(df.keys(), default_dict.keys()):
                self.assertEqual(df_key, dict_key)


if __name__ == "__main__":
    unittest.main(exit=False)
