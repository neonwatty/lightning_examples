import sys
import os
import datasets
from huggingface_hub import HfApi, upload_file
import argparse


def update_dataset_card(dataset_id: str, update: str, cache_dir="./dataset"):
    # Authenticate
    api = HfApi()

    # Download the README.md file
    readme_path = api.hf_hub_download(repo_id=dataset_id, filename="README.md", repo_type="dataset", cache_dir=cache_dir)

    # Read the existing content
    with open(readme_path, "r", encoding="utf-8") as f:
        dataset_card = f.read()

    # Update the content
    updated_card = dataset_card + "\n\n" + update

    # Save the updated content
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(updated_card)

    # Upload the updated dataset card
    upload_file(path_or_fileobj="README.md", path_in_repo="README.md", repo_id=dataset_id, repo_type="dataset", commit_message="Updated dataset card")


def create_sample_dataset(full_dataset_name, subset_name, sample_count=100, username="neonwatty", cache_dir="./dataset"):
    # Create a directory to save the sampled dataset
    os.makedirs(cache_dir, exist_ok=True)

    # Get the dataset name
    dataset_name = full_dataset_name.split("/")[-1]
    dataset_name_sample = f"{dataset_name}-sample-{sample_count}"

    # Load the dataset
    dataset = datasets.load_dataset(full_dataset_name, subset_name, cache_dir=cache_dir)

    # Get names of all splits
    splits = list(dataset.keys())

    # Sample 100 rows from the training split (or modify for other splits)
    for split in splits:
        # Collect sample
        split_sample = dataset[split].shuffle(seed=42).select(range(sample_count))

        # Push to hub
        split_sample.push_to_hub(dataset_name_sample, subset_name, split=split)
        print(f"INFO: {split} split pushed to the hub successfully")

    # Update the dataset card
    update = f"""
    # {dataset_name_sample}
    Sample of {sample_count} rows from the {full_dataset_name} dataset.
    """
    update_dataset_card(username + "/" + dataset_name_sample, update, cache_dir)
    print("INFO: Dataset card updated successfully")

    # Print url of dataset
    print(f"INFO: Dataset URL: https://huggingface.co/{username}/{dataset_name_sample}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset arguments.")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("--subset_name", type=str, default=None, help="Optional subset name (default: None)")
    parser.add_argument("--sample_count", type=int, default=50, help="Number of samples to process (default: 50)")

    args = parser.parse_args()

    dataset_name = args.dataset_name
    subset_name = args.subset_name
    sample_count = args.sample_count

    print(f"INFO: Generating sample for dataset: {dataset_name}, Subset: {subset_name}, Sample Count: {sample_count}")

    create_sample_dataset(dataset_name, subset_name, sample_count)
