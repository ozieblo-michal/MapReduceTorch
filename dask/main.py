import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import optuna
from transformers import (
    DistilBertTokenizer,
    DistilBertForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset, DatasetDict
import torch
from torch.utils.data import DataLoader

cluster = LocalCluster()

client = Client(cluster)

augmented_data_path = '/Users/michalozieblo/Desktop/mapreducetorch/scraper/augmented/data'

df = dd.read_parquet(augmented_data_path)

def dask_to_datasets(df):

    df_pd = df.compute()
    dataset = Dataset.from_pandas(df_pd)
    
    return dataset

dataset = dask_to_datasets(df)

train_test_split = dataset.train_test_split(test_size=0.1)

train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

train_df = train_dataset
eval_df = test_dataset


