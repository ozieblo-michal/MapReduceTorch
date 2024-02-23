from transformers import (
    DistilBertTokenizer,
    DistilBertForMaskedLM,
    pipeline,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

import os

BEST_MODEL_PATH = "./training/results_trial_1"

def find_latest_checkpoint(base_path):
    checkpoint_dirs = [d for d in os.listdir(base_path) if d.startswith('checkpoint')]
    latest_checkpoint = None
    latest_time = 0
    
    for checkpoint_dir in checkpoint_dirs:
        full_path = os.path.join(base_path, checkpoint_dir)
        stat = os.stat(full_path)
        if stat.st_mtime > latest_time:
            latest_checkpoint = checkpoint_dir
            latest_time = stat.st_mtime
            
    return latest_checkpoint


latest_checkpoint_dir = find_latest_checkpoint(BEST_MODEL_PATH)

if latest_checkpoint_dir:
    print(f"Latest checkpoint directory: {latest_checkpoint_dir}")
    print(f"Model path: {BEST_MODEL_PATH}")
    model_path = os.path.join(BEST_MODEL_PATH, latest_checkpoint_dir)
    model = DistilBertForMaskedLM.from_pretrained(model_path)
else:
    print("No checkpoint directories found.")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


def prepare_test_dataset(test_file_path: str):
    dataset = load_dataset("text", data_files={"test": test_file_path})
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        ),
        batched=True,
    )
    return tokenized_dataset["test"]


def evaluate_model(trained_model, test_dataset):
    training_args = TrainingArguments(
        output_dir="./results", per_device_eval_batch_size=8
    )
    trainer = Trainer(
        model=trained_model,
        args=training_args,
        eval_dataset=test_dataset,
    )
    return trainer.evaluate()


test_file_path = "test_data.txt"
test_dataset = prepare_test_dataset(test_file_path)
evaluation_result = evaluate_model(model, test_dataset)
print("Evaluation results:", evaluation_result)

fill_mask_pipeline = pipeline("fill-mask", model=model, tokenizer=tokenizer)

examples = ["sentence1 [MASK]", "sentence2 [MASK]"]

for example in examples:
    predictions = fill_mask_pipeline(example)
    print("\nInput:", example)
    for prediction in predictions:
        print(
            f"Prediction: {prediction['sequence']} (score: {prediction['score']:.4f})"
        )
