import torch
import datasets
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchvision import transforms
from tokenizers import trainers, Tokenizer
import os
from torch.utils.data import DataLoader, random_split


class HuggingFaceDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        source_tokenizer,
        target_tokenizer,
        source_lang="source_text",
        target_lang="target_text",
        max_length=512,
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
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_length = max_length

        # Load tokenizers from disk if available, otherwise train them
        self.source_tokenizer = self.load_or_train_tokenizer(self.source_tokenizer_path, self.source_lang)
        self.target_tokenizer = self.load_or_train_tokenizer(self.target_tokenizer_path, self.target_lang)

    def load_or_train_tokenizer(self, tokenizer_path, lang_key):
        """
        Load a tokenizer from disk if it exists, otherwise train and save it.
        """
        if os.path.exists(tokenizer_path):
            print(f"INFO: Loading tokenizer from {tokenizer_path}")
            return Tokenizer.from_file(tokenizer_path)
        else:
            print(f"INFO: Training tokenizer for {lang_key}...")
            tokenizer = self.train_tokenizer(lang_key)
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
        target_text = f"{self.target_tokenizer.bos_token} {item['translation'][self.target_lang]}"

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


class TranslationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        hf_dataset,
        source_lang="en",
        target_lang="es",
        batch_size=32,
        max_length=512,
        source_tokenizer_path="./tokenizers/source_tokenizer.json",
        target_tokenizer_path="./tokenizers/target_tokenizer.json",
        val_split=0.1,
        test_split=0.1,
    ):
        """
        Args:
            hf_dataset: A Hugging Face dataset object.
            source_lang (str): The key for the source language text.
            target_lang (str): The key for the target language text.
            batch_size (int): The batch size for DataLoader.
            max_length (int): Maximum sequence length for tokenization.
            source_tokenizer_path (str): Path to load/save the trained source tokenizer.
            target_tokenizer_path (str): Path to load/save the trained target tokenizer.
            val_split (float): Fraction of data to be used for validation.
            test_split (float): Fraction of data to be used for testing.
        """
        super().__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.val_split = val_split
        self.test_split = test_split

        # Initialize HuggingFaceDataset for training, validation, and testing
        self.train_dataset = HuggingFaceDataset(
            hf_dataset,
            source_lang=source_lang,
            target_lang=target_lang,
            max_length=max_length,
            source_tokenizer_path=source_tokenizer_path,
            target_tokenizer_path=target_tokenizer_path,
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
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
