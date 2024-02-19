import dask.dataframe as dd
from training import script
import pandas as pd
import torch
from unittest.mock import MagicMock
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

    item = dataset.__getitem__(0)
    
    assert isinstance(item, dict), "Retrieved item is not a dictionary."
    assert "input_ids" in item and isinstance(item["input_ids"], torch.Tensor), "Item does not contain input_ids or is not a tensor."
    assert "attention_mask" in item and isinstance(item["attention_mask"], torch.Tensor), "Item does not contain attention_mask or is not a tensor."
    assert "labels" in item and isinstance(item["labels"], torch.Tensor), "Item does not contain labels or is not a tensor."
    assert "token_type_ids" in item and isinstance(item["token_type_ids"], torch.Tensor), "Item does not contain token_type_ids or is not a tensor."
    assert torch.equal(item["input_ids"], torch.tensor([1, 2, 3], dtype=torch.long)), "input_ids do not match."
    assert torch.equal(item["attention_mask"], torch.tensor([1, 1, 1], dtype=torch.long)), "attention_mask do not match."
    assert torch.equal(item["labels"], torch.tensor([-100, 2, -100], dtype=torch.long)), "labels do not match."
    assert torch.equal(item["token_type_ids"], torch.tensor([0, 0, 0], dtype=torch.long)), "token_type_ids do not match."

def test_get_item_with_missing_fields():
    data = {
        'input_ids': [[1, 2, 3]],
        'attention_mask': [[1, 1, 1]]
    }
    ddf = dd.from_pandas(pd.DataFrame(data), npartitions=1)
    dataset = script.CustomDataset(ddf)
    item = dataset.__getitem__(0)
    
    assert item["labels"] is None, "'labels' should be None when missing."
    assert item["token_type_ids"] is None, "'token_type_ids' should be None when missing."
