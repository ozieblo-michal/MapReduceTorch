from scraper.main import convert_to_txt
from format.main import full_data_preparation_and_augmentation
from training.script import run_training_and_evaluation
from dask.distributed import Client
import boto3

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





def find_cluster_id_by_name(cluster_name):
    emr = boto3.client('emr')
    clusters = emr.list_clusters(ClusterStates=['STARTING', 'BOOTSTRAPPING', 'RUNNING', 'WAITING'])
    for cluster in clusters['Clusters']:
        if cluster['Name'] == cluster_name:
            return cluster['Id']
    return None

def get_master_node_dns(cluster_id):
    emr = boto3.client('emr')
    cluster = emr.describe_cluster(ClusterId=cluster_id)
    master_dns = cluster['Cluster']['MasterPublicDnsName']
    return master_dns


if __name__ == "__main__":
    cluster_name = "book2flash"
    cluster_id = find_cluster_id_by_name(cluster_name)
    if cluster_id:
        master_dns = get_master_node_dns(cluster_id)
        client = Client(f'tcp://{master_dns}:8786')
        run_training_and_evaluation(train_dataset_parquet_path, eval_dataset_parquet_path)
    else:
        print("Cannto find cluster name.")