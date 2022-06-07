# Test suite

The test suite can be run with the unittest framework.

To be able to run the data, you first need to download the test data.
The test data can be found in Azure storage and is managed and downloaded using [DVC](https://dvc.org/).

## Get the data

### Connection string

To download the dataset, you will need to get a connection_string to remotely access Azure Storage.
Head to [Azure portal](https://portal.azure.com/#home), and access resource groups.
In the page where you can see all resources make sure that `S_SDG_ADV_ANALYTICS_NONPROD` is selected.
You should see the resource group `RG_NT_ARCO_DEV`.
If the resource is not present you must be added to the Azure group, so go ping Malcolm.
In it, you will find the resource called `arcodatasets` containing all the data related to this repo.
Click on `access key``on the left panel and save your connection_string.

### Download

The default DVC remote needs to be configured to load the data.

> Setting the correct access in local or current repo does not work atm.
> Hence, we need to do in the global default.
> This is not great but for now we need to use this workaround.

First create a global remote called `azure` pointing to the container of all ARCo data: `dvc remote add -d --global azure azure://arcodatasets`.
Then, add your connection key to the remote `dvc remote modify --global azure connection_string CONNECTIONSTRING`, where `CONNECTIONSTRING` is your connection string found in the previous step.
Make sure the connection string is surrounded by `"`.

Running the script `python tests/data/load_test_data.py` should now load all necessary data for unit-testing.

## Run the test

Run the tests locally by running `python -m unittest discover tests` in the root directory.
