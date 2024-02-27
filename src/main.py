from scraper.main import convert_to_txt
from format.main import full_data_preparation_and_augmentation
from training.script import run_training_and_evaluation
from dask.distributed import Client

input_file_path = "./book2flash/src/files/source_text/input_text.epub"
output_file_path = "./book2flash/src/files/source_text/scrapped_text.txt"

input_file_path = "./book2flash/src/files/format/scrapped_text.txt"
filtered_output_file_path = "./book2flash/src/files/format/filtered_output.txt"
train_dataset_parquet_path = "./book2flash/src/files/format/train.parquet"
eval_dataset_parquet_path = "./book2flash/src/files/format/eval.parquet"

convert_to_txt(input_file_path, output_file_path)

full_data_preparation_and_augmentation(
        input_file_path,
        filtered_output_file_path,
        train_dataset_parquet_path,
        eval_dataset_parquet_path,
        augment_rate=0.3,
        n=2
    )

# client = Client()

# model = run_training_and_evaluation(train_dataset_parquet_path, eval_dataset_parquet_path)