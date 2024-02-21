import optuna
from datasets import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

cluster = LocalCluster()

client = Client(cluster)

augmented_data_path = (
    "/Users/michalozieblo/Desktop/mapreducetorch/scraper/augmented/data"
)

df = dd.read_parquet(augmented_data_path)

from datasets import Dataset, concatenate_datasets

from dask.distributed import as_completed


def process_partition(partition):
    df_pd = partition.compute()
    dataset = Dataset.from_pandas(df_pd)
    return dataset


def dask_to_datasets(df, client):

    futures = client.map(process_partition, df.to_delayed())

    datasets = []
    for future, result in as_completed(futures, with_results=True):
        datasets.append(result)

    combined_dataset = concatenate_datasets(datasets)

    return combined_dataset


dataset = dask_to_datasets(df, client)

train_test_split = dataset.train_test_split(test_size=0.1)

train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer.add_tokens(["CUDA", "GPU", "CPU", "DQP"])


def model_training_function(trial):

    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 5)
    per_device_train_batch_size = trial.suggest_categorical(
        "per_device_train_batch_size", [8, 16]
    )
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4)

    training_args = TrainingArguments(
        output_dir=f"./results_trial_{trial.number}",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        logging_dir="./logs",
        logging_steps=10,
        use_cpu=True,
    )

    model = DistilBertForMaskedLM.from_pretrained(
        "distilbert-base-uncased", return_dict=True
    )
    model.resize_token_embeddings(len(tokenizer))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        ),
    )

    trainer.train()
    eval_results = trainer.evaluate()

    return eval_results["eval_loss"]


import optuna

study = optuna.create_study(direction="minimize")
study.optimize(model_training_function, n_trials=3)

best_trial = study.best_trial
print(f"Best trial: {best_trial.number} with loss {best_trial.value}")

best_model_path = f"./xxxbest_model_trial_{best_trial.number}"
model = DistilBertForMaskedLM.from_pretrained(best_model_path)
