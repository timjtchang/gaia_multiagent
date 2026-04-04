import os
from datasets import load_dataset
from huggingface_hub import snapshot_download

# Define your local path
target_dir = "./data"

# Download everything to the local folder
data_dir = snapshot_download(
    repo_id="gaia-benchmark/GAIA", 
    repo_type="dataset",
    local_dir=target_dir,
    local_dir_use_symlinks=False
)

# Load the dataset from the local path
# Since GAIA has a loading script in the repo, load_dataset will find it in data_dir
dataset = load_dataset(data_dir, "2023_level1", split="test")

for example in dataset:
    question = example["Question"]
    # Now file_path will point to the files inside your ./gaia_data folder
    file_path = os.path.join(data_dir, example["file_path"])
    print(f"Question: {question[:50]}... -> File: {file_path}")