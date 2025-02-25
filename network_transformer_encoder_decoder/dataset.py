import torch
import datasets
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchvision import transforms
from tokenizers import trainers
from torch.utils.data import Dataset
import torch


class HuggingFaceDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        source_tokenizer,
        target_tokenizer,
        source_lang="source_text",
        target_lang="target_text",
        max_length=512,
        train_tokenizers=False,
        source_tokenizer_path="./tokenizers/source_tokenizer.json",
        target_tokenizer_path="./tokenizers/target_tokenizer.json",
    ):
        """
        Args:
            hf_dataset: A Hugging Face dataset object.
            source_tokenizer: Tokenizer for the source language.
            target_tokenizer: Tokenizer for the target language.
            source_lang (str): The key for the source language text.
            target_lang (str): The key for the target language text.
            max_length (int): Maximum sequence length for tokenization.
        """
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_length = max_length

        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

        # Train tokenizers if requested
        if train_tokenizers:
            self.train_tokenizers()

    def train_tokenizers(self):
        """
        Train the source and target tokenizers on the dataset text using Hugging Face tokenizers
        and save the trained tokenizers to disk.
        """
        print("Training tokenizers on dataset...")

        source_texts = [item["translation"][self.source_lang] for item in self.dataset]
        target_texts = [item["translation"][self.target_lang] for item in self.dataset]

        # Train the source tokenizer
        source_trainer = trainers.BpeTrainer(vocab_size=32000, special_tokens=["[PAD]", "[BOS]", "[EOS]"])
        self.source_tokenizer.train_from_iterator(source_texts, trainer=source_trainer)
        self.source_tokenizer.save(self.source_tokenizer_path)  # Save the trained tokenizer

        # Train the target tokenizer
        target_trainer = trainers.BpeTrainer(
            vocab_size=32000,
            special_tokens=["[PAD]", "[BOS]"],  # no eos for target
        )
        self.target_tokenizer.train_from_iterator(target_texts, trainer=target_trainer)
        self.target_tokenizer.save(self.target_tokenizer_path)  # Save the trained tokenizer

        print(f"Tokenizers trained and saved to {self.source_tokenizer_path} and {self.target_tokenizer_path}.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Format text with special tokens
        source_text = f"{self.source_tokenizer.bos_token} {item['translation'][self.source_lang]} {self.source_tokenizer.eos_token}"
        target_text = f"{self.target_tokenizer.bos_token} {item['translation'][self.target_lang]} {self.target_tokenizer.eos_token}"

        # Tokenize source text
        source_tokens = self.source_tokenizer.encode(source_text)
        target_tokens = self.target_tokenizer.encode(target_text)

        # Ensure token lengths are within the maximum length
        source_tokens = source_tokens.ids[: self.max_length]
        target_tokens = target_tokens.ids[: self.max_length]

        # Return as a dictionary compatible with Hugging Face models
        return {
            "input_ids": torch.tensor(source_tokens),
            "src_mask": torch.tensor([1] * len(source_tokens)),  # Attention mask for input
            "labels": torch.tensor(target_tokens),
            "tgt_mask": torch.tensor([1] * len(target_tokens)),  # Attention mask for target
        }


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_name="mnist", batch_size=64, sample_size=None, cache_dir="./dataset", num_workers=0):
        """
        :param dataset_name: Name of the Hugging Face dataset (e.g., "mnist").
        :param batch_size: Number of samples per batch.
        :param sample_size: Number of samples to use (if None, use full dataset).
        :param cache_dir: Directory to store cached datasets.
        :param num_workers: Number of worker threads for data loading.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.cache_dir = cache_dir
        self.num_workers = num_workers  # Added num_workers
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
            ]
        )

    def prepare_data(self):
        # Downloads the dataset (only needed once), specifying cache directory
        datasets.load_dataset(self.dataset_name, cache_dir=self.cache_dir)

    def setup(self, stage=None):
        # Load the dataset from cache
        dataset = datasets.load_dataset(self.dataset_name, cache_dir=self.cache_dir)

        # Apply sampling if sample_size is specified
        if self.sample_size:
            train_data = dataset["train"].select(range(min(self.sample_size, len(dataset["train"]))))
            val_data = dataset["test"].select(range(min(self.sample_size // 10, len(dataset["test"]))))
            test_data = dataset["test"].select(range(min(self.sample_size // 10, len(dataset["test"]))))
        else:
            train_data = dataset["train"]
            val_data = dataset["test"]
            test_data = dataset["test"]

        # Wrap in custom dataset class
        self.train_dataset = HuggingFaceDataset(train_data, transform=self.transform)
        self.val_dataset = HuggingFaceDataset(val_data, transform=self.transform)
        self.test_dataset = HuggingFaceDataset(test_data, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    @property
    def shapes(self):
        # Return the shapes of the dataset
        x, y = self.train_dataset[0]

        # flatten input_shape into input_size
        input_size = torch.flatten(x).shape[0]
        num_classes = 10
        return {"input_size": input_size, "num_classes": num_classes}
