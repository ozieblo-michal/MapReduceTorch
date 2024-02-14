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
    """
    Fetches synonyms for a given word.

    Args:
    - word (str): The word for which to find synonyms.

    Returns:
    - list: A list of synonyms for the given word.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lem in syn.lemmas():
            synonym = lem.name().replace("_", " ")
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def synonym_replacement(sentence, n=1):
    """
    Replaces up to n words in the sentence with their synonyms.

    Args:
    - sentence (str): The sentence to augment.
    - n (int): The maximum number of words to replace.

    Returns:
    - str: The augmented sentence.
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




spark = SparkSession.builder.appName("DistilBERT Training").getOrCreate()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


new_special_tokens = ["CUDA", "GPU", "CPU", "DQP"]

tokenizer.add_tokens(new_special_tokens)


from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType

def filter_by_token_count(sentence, max_tokens=512):
    tokens = tokenizer.tokenize(sentence)
    return len(tokens) <= max_tokens

filter_udf = udf(filter_by_token_count, BooleanType())

# Load data as DataFrame
data_df = spark.read.text("/Users/michalozieblo/Desktop/mapreducetorch/scraper/output/output.txt")

# Filter sentences by token limit using UDF
filtered_data_df = data_df.filter(filter_udf(data_df.value))

# Save or continue processing as needed
filtered_data_df.write.text("/Users/michalozieblo/Desktop/mapreducetorch/scraper/output/filtered_output.txt")

data = data_df


# Define UDF for data augmentation
def augment_sentence(sentence, augment_rate=0.1, n=1):
    # Tu możesz wykorzystać istniejącą logikę augmentacji, np. synonym_replacement
    return synonym_replacement(sentence, n) if random.uniform(0, 1) < augment_rate else sentence

augment_udf = udf(augment_sentence)

# Apply augmentation to DataFrame
augmented_data_df = filtered_data_df.withColumn("augmented", augment_udf(filtered_data_df.value))

augmented_data_df.write.mode('overwrite').parquet('/Users/michalozieblo/Desktop/mapreducetorch/scraper/augmented/data')


train_dataset, eval_dataset = augmented_data_df.randomSplit([0.9, 0.1])





# from dask.distributed import Client, LocalCluster

# cluster = LocalCluster()
# client = Client(cluster)





def model_training_function(trial):
    """
    Objective function for Optuna optimization. Trains the model and returns the evaluation loss.

    Args:
    - trial (optuna.trial.Trial): An Optuna trial object.

    Returns:
    - float: The evaluation loss of the model.
    """

    model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

    # IMPORTANT: Adjust the model's token embeddings to accommodate new tokens

    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=f"./results_trial_{trial.number}",
        num_train_epochs=trial.suggest_int("num_train_epochs", 1, 5),
        per_device_train_batch_size=trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16]
        ),
        learning_rate=trial.suggest_float("learning_rate", 5e-5, 5e-4),
        logging_dir="./logs",
        logging_steps=10,
        use_cpu=True,
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

    print(eval_results)

    model.save_pretrained(f"./best_model_trial_{trial.number}")

    return eval_results["eval_loss"]


# Run Optuna optimization
study = optuna.create_study(direction="minimize")
study.optimize(model_training_function, n_trials=3)

best_trial = study.best_trial
print(f"Best trial: {best_trial.number} with loss {best_trial.value}")

# Load the best model
best_model_path = f"./best_model_trial_{best_trial.number}"
model = DistilBertForMaskedLM.from_pretrained(best_model_path)
