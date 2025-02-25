import os
import torch
import datasets
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchvision import transforms
from tokenizers import trainers, Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from torch.utils.data import DataLoader, random_split
from network_transformer_encoder_decoder.config import DataConfig


class HuggingFaceDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        subset_name,
        cache_dir="cache/",
        source_lang="source_text",
        target_lang="target_text",
        max_length=512,
    ):
        """
        Args:
            dataset_name: The name of the Hugging Face dataset to use.
            subset_name: The name of the subset of the dataset to use.
            source_lang (str): The key for the source language text.
            target_lang (str): The key for the target language text.
            max_length (int): Maximum sequence length for tokenization.
        """
        # Create cache directories if they don't exist
        os.makedirs(cache_dir + "tokenizers/", exist_ok=True)
        os.makedirs(cache_dir + "dataset/", exist_ok=True)

        # load in the dataset
        self.dataset = datasets.load_dataset(dataset_name, subset_name, split="train", cache_dir=cache_dir + "dataset/")

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_length = max_length

        # Tokenizer paths
        self.source_tokenizer_path = cache_dir + "tokenizers/" + "source_tokenizer_{}.json".format(source_lang)
        self.target_tokenizer_path = cache_dir + "tokenizers/" + "target_tokenizer_{}.json".format(target_lang)

        # Load tokenizers from disk if available, otherwise train them
        self.source_tokenizer = self.load_or_train_tokenizer(cache_dir + "tokenizers/" + "source_tokenizer_{}.json".format(source_lang), source_lang)
        self.target_tokenizer = self.load_or_train_tokenizer(cache_dir + "tokenizers/" + "target_tokenizer_{}.json".format(target_lang), target_lang)

    def load_or_train_tokenizer(self, tokenizer_path, lang_key):
        """
        Load a tokenizer from disk if it exists, otherwise train and save it.
        """
        if os.path.exists(tokenizer_path):
            print(f"INFO: Loading tokenizer from {tokenizer_path}")
            return Tokenizer.from_file(tokenizer_path)
        else:
            print(f"INFO: Training tokenizer for {lang_key}...")
            tokenizer = self.train_tokenizers()
            tokenizer.save(tokenizer_path)  # Save the trained tokenizer
            return tokenizer

    def train_tokenizers(self):
        """
        Train the source and target tokenizers on the dataset text using Hugging Face tokenizers
        and save the trained tokenizers to disk.
        """
        print("Training tokenizers on dataset...")

        source_texts = [item["translation"][self.source_lang] for item in self.dataset]
        target_texts = [item["translation"][self.target_lang] for item in self.dataset]

        # Train the source tokenizer
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()
        source_trainer = trainers.BpeTrainer(vocab_size=32000, special_tokens=["[PAD]", "[BOS]", "[EOS]"])
        tokenizer.train_from_iterator(source_texts, trainer=source_trainer)
        tokenizer.save(self.source_tokenizer_path)  # Save the trained tokenizer

        # Train the target tokenizer
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()
        target_trainer = trainers.BpeTrainer(
            vocab_size=32000,
            special_tokens=["[PAD]", "[BOS]"],  # no eos for target
        )
        tokenizer.train_from_iterator(target_texts, trainer=target_trainer)
        tokenizer.save(self.target_tokenizer_path)  # Save the trained tokenizer

        print(f"Tokenizers trained and saved to {self.source_tokenizer_path} and {self.target_tokenizer_path}.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Format text with special tokens
        source_text = "[BOS]" + " " + item["translation"][self.source_lang] + " " + "[EOS]"
        target_text = "[BOS]" + " " + item["translation"][self.target_lang]

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
    def __init__(
        self,
        dataset_config: DataConfig,
        batch_size=64,
        val_split=0.1,
        test_split=0.1,
    ):
        """
        Args:
            dataset_config (DataConfig): Configuration for the dataset.
            batch_size (int): Batch size for training.
            val_split (float): Fraction of data to be used for validation.
            test_split (float): Fraction of data to be used for testing.
        """
        super().__init__()
        # unpack dataset config
        cache_dir = dataset_config.cache_dir
        dataset_name = dataset_config.dataset_name
        subset_name = dataset_config.dataset_subset
        source_lang = dataset_config.source_lang
        target_lang = dataset_config.target_lang
        max_seq_len = dataset_config.max_seq_len
        self.num_workers = dataset_config.num_workers
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split

        # Initialize HuggingFaceDataset for training, validation, and testing
        self.train_dataset = HuggingFaceDataset(
            dataset_name, subset_name, source_lang=source_lang, target_lang=target_lang, max_length=max_seq_len, cache_dir=cache_dir
        )

        # loop over dataset and print the length of each element: input_ids, src_mask, labels, tgt_mask
        for i in range(len(self.train_dataset)):
            print(
                len(self.train_dataset[i]["input_ids"]),
                len(self.train_dataset[i]["src_mask"]),
                len(self.train_dataset[i]["labels"]),
                len(self.train_dataset[i]["tgt_mask"]),
            )

        # Split dataset into training, validation, and test sets
        total_len = len(self.train_dataset)
        val_len = int(total_len * val_split)
        test_len = int(total_len * test_split)
        train_len = total_len - val_len - test_len

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.train_dataset, [train_len, val_len, test_len])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
