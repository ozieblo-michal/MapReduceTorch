import os

import optuna
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)

import dask.dataframe as dd
from dask.distributed import Client

os.environ["DASK_MULTIPROCESSING_METHOD"] = "spawn"

if __name__ == "__main__":

    client = Client()

    train_ddf = dd.read_parquet(
        "/Users/michalozieblo/Desktop/mapreducetorch/dask/augmented_parquet/train.parquet"
    )
    eval_ddf = dd.read_parquet(
        "/Users/michalozieblo/Desktop/mapreducetorch/dask/augmented_parquet/eval.parquet"
    )

    class CustomDataset(Dataset):
        def __init__(self, ddf):
            self.ddf = ddf.compute()

        def __len__(self):
            return len(self.ddf)

        def __getitem__(self, idx):
            row = self.ddf.iloc[idx]

            input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
            token_type_ids = (
                torch.tensor(row["token_type_ids"], dtype=torch.long)
                if "token_type_ids" in row
                else None
            )
            attention_mask = torch.tensor(row["attention_mask"], dtype=torch.long)
            labels = (
                torch.tensor(row["labels"], dtype=torch.long)
                if "labels" in row
                else None
            )

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "token_type_ids": token_type_ids,
            }

    train_dataset = CustomDataset(train_ddf)
    eval_dataset = CustomDataset(eval_ddf)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=True, num_workers=0)

    def model_training_function(trial):
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        new_special_tokens = ["CUDA", "GPU", "CPU", "DQP"]
        tokenizer.add_tokens(new_special_tokens)

        num_train_epochs = trial.suggest_int("num_train_epochs", 1, 5)
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4)

        model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
        model.resize_token_embeddings(len(tokenizer))

        training_args = TrainingArguments(
            output_dir=f"./results_trial_{trial.number}",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=8,
            learning_rate=learning_rate,
            logging_dir="./logs",
        )

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

    study = optuna.create_study(direction="minimize")
    study.optimize(model_training_function, n_trials=3)

    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number} with loss {best_trial.value}")

    best_model_path = f"./best_model_trial_{best_trial.number}"
    model = DistilBertForMaskedLM.from_pretrained(best_model_path)
