import random
import nltk
from nltk.corpus import wordnet
from pyspark.sql import SparkSession
from transformers import DistilBertTokenizer
from datasets import Dataset
import torch
import numpy as np

import boto3

import os

bucket_name = os.getenv('S3_BUCKET_NAME')



INPUT_TEXT_FILE_PATH = "./scraper/output/output.txt"
FILTERED_OUTPUT_FILE_PATH = "./scraper/output/filtered_output.txt"
TRAIN_DATASET_PARQUET_PATH = "./training/augmented_parquet/train.parquet"
EVAL_DATASET_PARQUET_PATH = "./training/augmented_parquet/eval.parquet"



def download_file_from_s3(bucket_name, object_name, local_file_name):
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, object_name, local_file_name)

def upload_file_to_s3(local_file_name, bucket_name, object_name):
    s3 = boto3.client('s3')
    s3.upload_file(local_file_name, bucket_name, object_name)



def download_nltk_resource(resource_name: str, download_dir: str = None):
    """
    Checks if an NLTK resource is available locally; if not, downloads it.

    Args:
    - resource_name (str): The name of the NLTK resource.
    - download_dir (str, optional): The directory to download the NLTK resource to.
                                    If None, uses NLTK's default download directory.
    """
    try:
        # Attempt to find the resource. If found, this does nothing.
        nltk.data.find(resource_name)
    except LookupError:
        nltk.download(resource_name.split("/")[1], download_dir=download_dir)


# Download necessary NLTK resources for finding synonyms
download_nltk_resource("corpora/wordnet")


def get_synonyms(word: str) -> list[str]:
    """
    Finds and returns a list of synonyms for a given word using NLTK's WordNet.

    Args:
    - word (str): The word for which synonyms are to be found.

    Returns:
    - list[str]: A list of unique synonyms for the word.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lem in syn.lemmas():
            synonym = lem.name().replace("_", " ")
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def synonym_replacement(sentence: str, n: int = 1) -> str:
    """
    Replaces up to 'n' words in the given sentence with their synonyms.

    Args:
    - sentence (str): The original sentence.
    - n (int): The maximum number of words to replace with synonyms.

    Returns:
    - str: The modified sentence with up to 'n' synonyms replaced.
    """
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


def augment_example(example: dict, augment_rate: float = 0.1, n: int = 1) -> dict:
    """
    Augments a text example by replacing up to 'n' words with synonyms based on a given augmentation rate.

    Args:
    - example (dict): The example to augment, containing at least a "text" key.
    - augment_rate (float): The probability of applying augmentation to the text.
    - n (int): The maximum number of words to replace with synonyms in the text.

    Returns:
    - dict: A dictionary with the key "text" containing the augmented text.
    """
    augmented_text = (
        synonym_replacement(example["text"], n=n)
        if random.uniform(0, 1) < augment_rate
        else example["text"]
    )
    return {"text": augmented_text}





def filter_sentences_by_token_limit(
    input_file_path: str, output_file_path: str, max_tokens: int = 512
) -> None:
    """
    Filters sentences from an input file and writes those with a token count less than or equal to 'max_tokens' to an output file.

    Args:
    - input_file_path (str): Path to the input file.
    - output_file_path (str): Path to the output file where filtered sentences will be written.
    - max_tokens (int): The maximum number of tokens allowed for a sentence to be included in the output.
    """
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    with open(input_file_path, "r", encoding="utf-8") as input_file, open(
        output_file_path, "w", encoding="utf-8"
    ) as output_file:
        for line in input_file:
            tokens = tokenizer.tokenize(line)
            if len(tokens) <= max_tokens:
                output_file.write(line)


def improved_masking(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: DistilBertTokenizer,
    mask_probability: float = 0.15,
):
    """
    Applies an improved masking strategy to input_ids based on mask_probability.

    Args:
    - input_ids (torch.Tensor): Tensor of token ids.
    - attention_mask (torch.Tensor): Tensor indicating which tokens should be attended to.
    - tokenizer (DistilBertTokenizer): Tokenizer instance used for token id conversions.
    - mask_probability (float): Probability with which a token will be masked.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Modified input_ids, unchanged attention_mask, and labels for training.
    """
    labels = input_ids.clone()
    candidates = (
        (input_ids != tokenizer.cls_token_id)
        & (input_ids != tokenizer.sep_token_id)
        & (input_ids != tokenizer.pad_token_id)
        & (torch.rand(input_ids.shape) < mask_probability)
    )

    selection = torch.where(candidates)
    for i in range(selection[0].size(0)):
        random_action = np.random.choice(
            ["mask", "random_token", "unchanged"], p=[0.8, 0.1, 0.1]
        )
        if random_action == "mask":
            input_ids[selection[0][i], selection[1][i]] = tokenizer.mask_token_id
        elif random_action == "random_token":
            input_ids[selection[0][i], selection[1][i]] = torch.randint(
                0, len(tokenizer), size=(1,)
            )

    labels[~candidates] = -100
    return input_ids, attention_mask, labels



def full_data_preparation_and_augmentation(input_file_path, filtered_output_file_path, train_dataset_parquet_path, eval_dataset_parquet_path, augment_rate, n, bucket_name):

    spark = SparkSession.builder.appName("DistilBERT Training").getOrCreate()

    download_file_from_s3(bucket_name, INPUT_TEXT_FILE_PATH, INPUT_TEXT_FILE_PATH)


    filter_sentences_by_token_limit(input_file_path, filtered_output_file_path)
    data = spark.read.text(filtered_output_file_path).rdd.map(lambda r: r[0]).collect()

    data = Dataset.from_dict({"text": data})
    train_data, eval_data = data.train_test_split(test_size=0.1).values()

    def augment_data(dataset):
        return dataset.map(lambda example: augment_example(example, augment_rate=augment_rate, n=n))

    train_augmented_data = augment_data(train_data)
    eval_augmented_data = augment_data(eval_data)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer.add_tokens(["CUDA", "GPU", "CPU", "DQP"])

    def tokenize_and_mask_function(examples):
        return tokenizer(examples["text"], max_length=512, padding="max_length", truncation=True)

    train_dataset = train_augmented_data.map(tokenize_and_mask_function, batched=True)
    eval_dataset = eval_augmented_data.map(tokenize_and_mask_function, batched=True)

    train_dataset.set_format(type='pandas').to_pandas().to_parquet(train_dataset_parquet_path)
    eval_dataset.set_format(type='pandas').to_pandas().to_parquet(eval_dataset_parquet_path)

    upload_file_to_s3(TRAIN_DATASET_PARQUET_PATH, bucket_name, TRAIN_DATASET_PARQUET_PATH)
    upload_file_to_s3(EVAL_DATASET_PARQUET_PATH, bucket_name, EVAL_DATASET_PARQUET_PATH)


if __name__ == "__main__":

    full_data_preparation_and_augmentation(
        input_file_path=INPUT_TEXT_FILE_PATH,
        filtered_output_file_path=FILTERED_OUTPUT_FILE_PATH,
        train_dataset_parquet_path=TRAIN_DATASET_PARQUET_PATH,
        eval_dataset_parquet_path=EVAL_DATASET_PARQUET_PATH,
        augment_rate=0.3,
        n=2
    )
