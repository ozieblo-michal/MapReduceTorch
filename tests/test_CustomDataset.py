import dask
dask.config.set({'dataframe.query-planning': True})
import dask.dataframe as dd
from training import script
import pandas as pd
import pytest

@pytest.fixture
def mock_parquet_data(tmp_path):
    data = {
        'input_ids': [[1, 2, 3]],
        'attention_mask': [[1, 1, 1]],
        'labels': [[-100, 2, -100]],
        'token_type_ids': [[0, 0, 0]]
    }
    df = pd.DataFrame({key: [value] for key, value in data.items()})
    temp_file = tmp_path / "temp.parquet"
    df.to_parquet(temp_file)
    return temp_file

def test_custom_dataset_initialization(mock_parquet_data):

    ddf = dd.read_parquet(mock_parquet_data)
    dataset = script.CustomDataset(ddf)
    
    assert len(dataset.ddf) == 1, "Dataset did not load data correctly."
    assert len(dataset) == 1, "Dataset length is incorrect."

