import optuna
import numpy as np
from transformers import DistilBertForMaskedLM, Trainer, TrainingArguments
from pyspark.sql import SparkSession
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset

import random

import nltk

nltk.download('wordnet')

from nltk.corpus import wordnet

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lem in syn.lemmas():
            synonym = lem.name().replace('_', ' ')
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n: 
            break

    sentence = ' '.join(new_words)
    return sentence



def augment_example(example, augment_rate=0.1, n=1):
    augmented_text = synonym_replacement(example['text'], n=n) if random.uniform(0, 1) < augment_rate else example['text']
    return {"text": augmented_text}



spark = SparkSession.builder \
    .appName("DistilBERT Training") \
    .getOrCreate()


def filter_sentences_by_token_limit(input_file_path: str, output_file_path: str, max_tokens: int = 512) -> None:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    with open(input_file_path, 'r', encoding='utf-8') as input_file, open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            tokens = tokenizer.tokenize(line)
            if len(tokens) <= max_tokens:
                output_file.write(line)

input_file_path = "/mapreducetorch/scraper/output/output.txt"
output_file_path = "/mapreducetorch/scraper/output/filtered_output.txt"
filter_sentences_by_token_limit(input_file_path, output_file_path)

data = spark.read.text("/mapreducetorch/scraper/output/filtered_output.txt").rdd.map(lambda r: r[0])
data = data.collect()  
data = Dataset.from_dict({'text': data})
augmented_data = data.map(lambda example: augment_example(example, augment_rate=0.3, n=2))
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
data = Dataset.from_dict({"text": augmented_data})

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_dataset = augmented_data.map(tokenize_function, batched=True)

def model_training_function(trial):
    model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
    training_args = TrainingArguments(
        output_dir=f'./results_trial_{trial.number}',
        num_train_epochs=trial.suggest_int("num_train_epochs", 1, 5),
        per_device_train_batch_size=trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
        learning_rate=trial.suggest_float("learning_rate", 5e-5, 5e-4),
        logging_dir='./logs',
        logging_steps=10,
        no_cuda=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15),
    )

    trainer.train()
    eval_results = trainer.evaluate()

    model.save_pretrained(f'./best_model_trial_{trial.number}')

    return eval_results["eval_loss"]

study = optuna.create_study(direction="minimize")
study.optimize(model_training_function, n_trials=3)

best_trial = study.best_trial
print(f"Best trial: {best_trial.number} with loss {best_trial.value}")

best_model_path = f'./best_model_trial_{best_trial.number}'

model = DistilBertForMaskedLM.from_pretrained(best_model_path)
