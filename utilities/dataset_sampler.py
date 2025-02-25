import sys
import os
import datasets
from huggingface_hub import HfApi, upload_file


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


def create_sample_dataset(full_dataset_name, sample_count=100, username="neonwatty", cache_dir="./dataset"):
    # Create a directory to save the sampled dataset
    os.makedirs(cache_dir, exist_ok=True)

    # Get the dataset name
    dataset_name = full_dataset_name.split("/")[-1]
    dataset_name_sample = f"{dataset_name}-sample-{sample_count}"

    # Load the dataset
    dataset = datasets.load_dataset(full_dataset_name, cache_dir=cache_dir)

    # Sample 100 rows from the training split (or modify for other splits)
    train_sample = dataset["train"].shuffle(seed=42).select(range(sample_count))
    test_sample = dataset["test"].shuffle(seed=42).select(range(sample_count))

    # Push to hub
    train_sample.push_to_hub(dataset_name_sample, split="train")
    print("INFO: Train split pushed to the hub successfully")

    test_sample.push_to_hub(dataset_name_sample, split="test")
    print("INFO: Test split pushed to the hub successfully")

    # Update the dataset card
    update = f"""
    # {dataset_name_sample}
    Sample of {sample_count} rows from the {full_dataset_name} dataset.
    """
    update_dataset_card(username + "/" + dataset_name_sample, update, cache_dir)
    print("INFO: Dataset card updated successfully")


if __name__ == "__main__":
    # unpack command line args
    args = sys.argv[1:]
    dataset_name = args[0]
    sample_count = 100
    if len(args) > 1:
        sample_count = int(args[1])
    create_sample_dataset(dataset_name, sample_count)
    print("INFO: Sample dataset created successfully")
