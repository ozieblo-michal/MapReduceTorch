# Import SparkSession to initialize a Spark session for distributed data processing.
# What for: To enable distributed data processing using Spark.

from pyspark.sql import SparkSession

# Import necessary classes from transformers library to work with pre-trained DistilBERT models and training utilities.
# What for: To use DistilBERT for masked language modeling and train the model.
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import pipeline

# Initialize a Spark session.
# What for: To create a Spark session for distributed data processing.
spark = SparkSession.builder \
    .appName("DistilBERT Training") \
    .getOrCreate()


def filter_sentences_by_token_limit(input_file_path: str, output_file_path: str, max_tokens: int = 512) -> None:
    """
    Filters out sentences from the input file that exceed the DistilBERT token limit and writes the rest to the output file.

    Parameters:
    - input_file_path (str): Path to the input text file.
    - output_file_path (str): Path to the output text file where sentences within the token limit will be saved.
    - max_tokens (int): Maximum number of tokens allowed per sentence. Defaults to 512.
    """
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    with open(input_file_path, 'r', encoding='utf-8') as input_file, open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            tokens = tokenizer.tokenize(line)
            if len(tokens) <= max_tokens:
                output_file.write(line)

input_file_path = "output.txt"
output_file_path = "filtered_output.txt"
filter_sentences_by_token_limit(input_file_path, output_file_path)

# Read text data from a file into a Spark DataFrame.
# What for: To load text data for training and evaluation.
data = spark.read.text("filtered_output.txt")


# Initialize a DistilBERT tokenizer.
# What for: To tokenize the text data for training the DistilBERT model.
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# This model is trained on text data where 
# the letter case has been normalized to lowercase (uncased). The tokenizer serves the purpose of splitting 
# text into tokens, the smallest units of text, which can then be passed to the model for further analysis 
# or prediction. In the case of DistilBERT, this tokenizer is compatible with the model and performs 
# tokenization operations consistent with how the model was trained.

# Tokenize the text data using the tokenizer.
# What for: To convert text data into input IDs suitable for the DistilBERT model.
tokenized_data = data.rdd.map(lambda row: tokenizer(row.value)["input_ids"])

# Find the maximum length of tokenized sequences.
# What for: To determine the maximum length of input sequences for padding.
max_length = tokenized_data.map(len).reduce(max)

# Pad tokenized sequences to the maximum length.
# What for: To ensure uniform length of input sequences required by the model.
padded_data = tokenized_data.map(lambda x: x + [0] * (max_length - len(x)))

# The second step involves aligning all tokenized sequences to the same length by adding the necessary 
# number of zero tokens to the end of each sequence. This is necessary because many natural language 
# processing models require all input data to have the same length. This enables efficient batch processing, 
# speeding up model training and utilizing computational resources such as GPUs. Padding does not affect 
# the informational content of the original text, as only zero tokens are added to the end of the sequences, 
# ensuring consistency and uniformity of input data for the model.

# For example, let's say we have a set of tokenized sentences as follows:

# Sentence 1: [101, 202, 303, 102]
# Sentence 2: [101, 404, 102
# In order to align these sequences to the same length, we would need to pad the second sentence with zero tokens, resulting in:

# Sentence 2 (padded): [101, 404, 102, 0]
# Now both sentences have the same length, allowing them to be efficiently processed by the model.

# Prepare training data as a list of dictionaries.
# What for: To convert padded data into a format acceptable for model training.
train_data = padded_data.map(lambda x: {"input_ids": x})
train_dataset = train_data.toDF(["input_ids"])

# Convert the training dataset into a list of dictionaries.
# What for: To convert the Spark DataFrame into a format suitable for model training.
train_data_list = train_dataset.collect()
train_data_dict_list = [row.asDict() for row in train_data_list]

# Initialize a DistilBERT model for masked language modeling.
# What for: To use the DistilBERT architecture for predicting masked tokens.
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

# Specify training arguments such as output directory, number of epochs, and batch size.
# What for: To configure the training process including where to save the model and how many epochs to train.
training_args = TrainingArguments(
    output_dir='./output',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    logging_dir='./logs',
    max_steps=500
)

# per_device_train_batch_size:
# Effect: Determines the number of training examples processed simultaneously on each device (e.g., GPU).
# Alternative: Changing the batch size per device, for example, per_device_train_batch_size=16.

# max_steps:
# Effect: Specifies the maximum number of training steps (batches) that will be executed during training.
# Alternative: Adjusting the maximum number of steps, for example, max_steps=500.

# Prepare data collator for language modeling.
# What for: To define how input data should be collated during training.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.20
)

# DataCollatorForLanguageModeling is a class used in the Hugging Face Transformers library to preprocess 
# input data before passing it to the model. In the context of natural language modeling, DataCollatorForLanguageModeling 
# is a specialized data collator designed specifically for language models.

# The main task of a data collator for natural language modeling is to prepare data for training the model 
# in a format required by a given model. In the case of natural language modeling, the data collator is responsible 
# for creating instances of a Masked Language Model (MLM).

# The parameters passed to DataCollatorForLanguageModeling have the following meanings:

# tokenizer: The tokenizer used to tokenize texts. It is used by the data collator to process input data.
# mlm=True: Specifies whether the language model is to be modeled as a Masked Language Model (MLM). When set 
# to True, some tokens in the input data will be masked, and the model will be trained to predict these tokens 
# based on context.
# mlm_probability=0.15: Specifies the probability of masking tokens in the input data. In this case, the masking 
# probability is 15%.

# Increasing or decreasing mlm_probability has the following consequences:

# Increasing mlm_probability: This would result in more tokens being masked in the input data during training. 
# As a consequence, the model would be trained to predict masked tokens more frequently, potentially leading to better 
# performance in tasks where understanding context and predicting missing words is important, but it could also increase 
# the difficulty of training and potentially lead to overfitting if set too high.
# Decreasing mlm_probability: This would result in fewer tokens being masked in the input data during training. Consequently, 
# the model would be trained less frequently on predicting masked tokens, which could potentially lead to worse performance 
# in tasks that rely heavily on understanding context and predicting missing words. However, it could make the training process 
# easier and less prone to overfitting.


# Initialize a Trainer object for model training.
# What for: To configure and execute the training process for the DistilBERT model.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data_dict_list,
    data_collator=data_collator,
)

# Train the DistilBERT model.
# What for: To update the model parameters based on the training data.
trainer.train()

# Save the trained model to the output directory.
# What for: To persist the trained model for future use.
model.save_pretrained("./output")

# Define a function to prepare the test dataset.
# What for: To tokenize and pad the test data for evaluation.
def prepare_test_dataset():
    test_data = spark.read.text("test_data.txt")
    tokenized_test_data = test_data.rdd.map(lambda row: tokenizer(row.value)["input_ids"])
    padded_test_data = tokenized_test_data.map(lambda x: x + [0] * (max_length - len(x)))
    test_data = padded_test_data.map(lambda x: {"input_ids": x}).toDF(["input_ids"])
    return test_data

# Define a function to evaluate the model on the test dataset.
# What for: To assess the performance of the trained model on unseen data.
def evaluate_model(trained_model, test_dataset):
    trainer = Trainer(
        model=trained_model,
        args=training_args,
        train_dataset=None,  
        data_collator=data_collator,  
    )
    
    test_data_list = test_dataset.collect()
    test_data_dict_list = [row.asDict() for row in test_data_list]
    
    eval_result = trainer.evaluate(test_data_dict_list)
    return eval_result

# Prepare the test dataset.
# What for: To tokenize and pad the test data for evaluation.
# test_dataset = prepare_test_dataset()

# Evaluate the model on the test dataset.
# What for: To assess the model's performance on unseen data.
# evaluation_result = evaluate_model(model, test_dataset)

# Print the evaluation results.
# What for: To display the performance metrics of the model.
# print("Evaluation results:", evaluation_result)

# loss: 0.2497 - This is the loss value achieved during the last training session. A lower loss value indicates better performance, meaning the model makes better predictions.
# learning_rate: 0.0 - Indicates the learning rate value at the end of the last epoch. A value of 0.0 indicates that the training has finished, as the learning rate has been reduced to zero.
# epoch: 250.0 - Indicates that the last training epoch was numbered 250.0. An epoch represents one iteration through the entire training dataset.
# train_runtime: 74.9566 - The duration of the training process, expressed in seconds.
# train_samples_per_second: 53.364 - The average number of samples processed per second during training. A higher value indicates a faster training process.
# train_steps_per_second: 6.671 - The average number of training steps processed per second. A training step is a single update of the model's weights based on a single batch of data.
# eval_loss: 2.5724399089813232 - The loss value obtained during the evaluation of the model on the test dataset. This value allows assessing how well the model generalizes to new data. A lower loss value indicates better performance.
# eval_runtime: 0.6892 - The duration of the evaluation process, expressed in seconds.
# eval_samples_per_second: 14.509 - The average number of samples processed per second during evaluation.
# eval_steps_per_second: 2.902 - The average number of evaluation steps processed per second.

# Define a pipeline for masked token prediction.
# What for: To generate predictions for masked tokens using the trained model.
fill_mask = pipeline(
    "fill-mask",
    model="./output",  # Path to the directory containing the trained model
    tokenizer="distilbert-base-uncased"
)

# Define examples with masked tokens.
# What for: To demonstrate how the model predicts masked tokens in sentences.
examples = [
    "I like to [MASK] on the weekends.",
    "She enjoys [MASK] books in her free time.",
    "The [MASK] is shining brightly today.",
    "Incorporating a [MASK] simplifies the management of workflows."
]

# Generate predictions for each example.
# What for: To demonstrate the model's ability to predict masked tokens in different contexts.
for example in examples:
    predictions = fill_mask(example)
    print("Input:", example)
    print("Predictions:", predictions)
    print()
