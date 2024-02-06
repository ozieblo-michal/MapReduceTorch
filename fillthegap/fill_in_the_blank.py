from transformers import DistilBertTokenizer, DistilBertForMaskedLM, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

# Loading the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    """
    Tokenizes the text examples, adding [MASK] tokens for masked language modeling.

    Args:
        examples: A dictionary with a key "text" containing the text examples.

    Returns:
        Tokenized and masked text examples.
    """
    # Replacing some words with the [MASK] token might be more complex and task-dependent
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Assuming we have a dataset `text_dataset` with a "text" column
text_dataset = load_dataset("text", data_files="/path/to/your/dataset.txt")['train']
tokenized_datasets = text_dataset.map(tokenize_function, batched=True)

# Data collator is used for dynamic word masking
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Loading the pre-trained DistilBertForMaskedLM model
model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

# Defining training hyperparameters
training_args = TrainingArguments(
    output_dir="./distilbert-finetuned-mlm",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# Initializing the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
)

# Starting the training process
trainer.train()

# Saving the trained model
trainer.save_model("./distilbert-finetuned-mlm")

# Loading the trained model and tokenizer
model_path = "./distilbert-finetuned-mlm"  # Path to the directory with the trained model
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForMaskedLM.from_pretrained(model_path)

# Preparing a sample sentence with a mask
text = "The capital of France is [MASK]."
input_ids = tokenizer.encode(text, return_tensors="pt")

# Generating predictions for the mask
with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs.logits

# Finding the token ID with the highest score for the mask
masked_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, masked_index].argmax(axis=1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"Original text: {text}")
print(f"Predicted fill-in: The capital of France is {predicted_token}.")
