import optuna
import random
import nltk
from nltk.corpus import wordnet
from pyspark.sql import SparkSession
from transformers import (
    DistilBertTokenizer,
    DistilBertForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

# Download necessary NLTK resources for finding synonyms
nltk.download("wordnet")


def get_synonyms(word):

    synonyms = set()
    for syn in wordnet.synsets(word):
        for lem in syn.lemmas():
            synonym = lem.name().replace("_", " ")
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

    sentence = " ".join(new_words)
    return sentence


def augment_example(example, augment_rate=0.1, n=1):

    augmented_text = (
        synonym_replacement(example["text"], n=n)
        if random.uniform(0, 1) < augment_rate
        else example["text"]
    )
    return {"text": augmented_text}


spark = SparkSession.builder.appName("DistilBERT Training").getOrCreate()


def filter_sentences_by_token_limit(
    input_file_path: str, output_file_path: str, max_tokens: int = 512
):

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    with open(input_file_path, "r", encoding="utf-8") as input_file, open(
        output_file_path, "w", encoding="utf-8"
    ) as output_file:
        for line in input_file:
            tokens = tokenizer.tokenize(line)
            if len(tokens) <= max_tokens:
                output_file.write(line)


input_file_path = (
    "/Users/michalozieblo/Desktop/mapreducetorch/scraper/output/output.txt"
)
output_file_path = (
    "/Users/michalozieblo/Desktop/mapreducetorch/scraper/output/filtered_output"
)
filter_sentences_by_token_limit(input_file_path, output_file_path)

data = spark.read.text(output_file_path).rdd.map(lambda r: r[0]).collect()
data = Dataset.from_dict({"text": data})

train_data, eval_data = data.train_test_split(test_size=0.1).values()

train_augmented_data = train_data.map(
    lambda example: augment_example(example, augment_rate=0.3, n=2)
)
eval_augmented_data = eval_data.map(
    lambda example: augment_example(example, augment_rate=0.3, n=2)
)







tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


new_special_tokens = ["CUDA", "GPU", "CPU", "DQP"]

tokenizer.add_tokens(new_special_tokens)


# def tokenize_function(examples):
#     """
#     Tokenizes the texts from examples.

#     Args:
#     - examples (dict): A dictionary containing the key 'text' with a list of texts to tokenize.

#     Returns:
#     - dict: A dictionary containing tokenized texts.
#     """
#     return tokenizer(
#         examples["text"], padding="max_length", truncation=True, max_length=512
#     )


import torch

import torch
from transformers import BertTokenizer
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.encode_plus("The quick brown fox jumps over the lazy dog", max_length=512, padding='max_length', truncation=True, return_tensors='pt')

input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']

def improved_masking(input_ids, attention_mask, tokenizer, mask_probability=0.15):
    labels = input_ids.clone()
    candidates = (input_ids != tokenizer.cls_token_id) & \
                 (input_ids != tokenizer.sep_token_id) & \
                 (input_ids != tokenizer.pad_token_id) & \
                 (torch.rand(input_ids.shape) < mask_probability)

    selection = torch.where(candidates)
    for i in range(selection[0].size(0)):
        random_action = np.random.choice(["mask", "random_token", "unchanged"], p=[0.8, 0.1, 0.1])
        if random_action == "mask":
            input_ids[selection[0][i], selection[1][i]] = tokenizer.mask_token_id
        elif random_action == "random_token":
            input_ids[selection[0][i], selection[1][i]] = torch.randint(0, len(tokenizer), size=(1,))

    labels[~candidates] = -100
    return input_ids, attention_mask, labels

input_ids, attention_mask, labels = improved_masking(input_ids, attention_mask, tokenizer)

print("Input IDs:", input_ids)
print("Labels:", labels)


from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_and_mask_function(examples):
    inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")

    input_ids, attention_mask, labels = improved_masking(inputs['input_ids'], inputs['attention_mask'], tokenizer)

    inputs['input_ids'] = input_ids
    inputs['attention_mask'] = attention_mask
    inputs['labels'] = labels

    return inputs

from datasets import Dataset

data = {'text': ["The quick brown fox jumps over the lazy dog", "I love machine learning"]}
dataset = Dataset.from_dict(data)

tokenized_dataset = dataset.map(tokenize_and_mask_function, batched=True)

for i in range(len(tokenized_dataset)):
    print(tokenized_dataset[i])

train_dataset = train_augmented_data.map(tokenize_and_mask_function, batched=True)
eval_dataset = eval_augmented_data.map(tokenize_and_mask_function, batched=True)

train_dataset = train_dataset.to_pandas()
eval_dataset = eval_dataset.to_pandas()

train_dataset.to_parquet('/Users/michalozieblo/Desktop/mapreducetorch/dask/augmented_parquet/train.parquet')
eval_dataset.to_parquet('/Users/michalozieblo/Desktop/mapreducetorch/dask/augmented_parquet/eval.parquet')
