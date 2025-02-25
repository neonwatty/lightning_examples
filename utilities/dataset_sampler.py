import os
from datasets import load_dataset, Dataset, DatasetDict
import numpy as np


def create_sample_dataset(full_dataset_name, sample_count=100, save_dir="./dataset"):
    # Create a directory to save the sampled dataset
    os.makedirs(save_dir, exist_ok=True)

    # Get the dataset name
    dataset_name = full_dataset_name.split("/")[-1]
    dataset_name_sample = f"{dataset_name}-sample-{sample_count}"
    full_dataset_name_sample = f"neonwatty/{dataset_name_sample}"

    # Load the dataset
    dataset = datasets.load_dataset(full_dataset_name, cache_dir="./dataset")

    # Sample 100 rows from the training split (or modify for other splits)
    train_sample = dataset["train"].shuffle(seed=42).select(range(sample_count))
    test_sample = dataset["test"].shuffle(seed=42).select(range(sample_count))

    # push to hub
    train_sample.push_to_hub(dataset_name_sample, split="train")
    test_sample.push_to_hub(dataset_name_sample, split="test")


dataset_name = "ylecun/mnist"
create_sample_dataset(dataset_name)
