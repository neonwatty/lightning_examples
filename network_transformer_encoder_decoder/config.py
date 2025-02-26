import os
import torch
from dataclasses import dataclass


@dataclass
class ModelDimensions:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int
    max_seq_len: int
    n_head: int
    n_layers: int
    dropout: float


@dataclass
class DataConfig:
    cache_dir: str
    num_workers: int
    dataset_name: str
    dataset_subset: str
    source_lang: str
    target_lang: str
    max_seq_len: int


# Training hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 1
CACHE_DIR = "cache/"

# Create data configurations
DATA_CONFIG_TEST = DataConfig(
    cache_dir=CACHE_DIR,
    num_workers=os.cpu_count(),
    dataset_name="neonwatty/opus_books-sample-50",
    dataset_subset="en-es",
    source_lang="en",
    target_lang="es",
    max_seq_len=512,
)

DATA_CONFIG = DataConfig(
    cache_dir=CACHE_DIR,
    num_workers=os.cpu_count(),
    dataset_name="Helsinki-NLP/opus_books",
    dataset_subset="en-es",
    source_lang="en",
    target_lang="es",
    max_seq_len=512,
)


# setup model configurations
MODEL_CONFIG_TEST = ModelDimensions(
    src_vocab_size=3200,
    tgt_vocab_size=3200,
    d_model=768,
    max_seq_len=512,
    n_head=8,
    n_layers=2,
    dropout=0.1,
)

MODEL_CONFIG = ModelDimensions(
    src_vocab_size=3200,
    tgt_vocab_size=3200,
    d_model=768,
    max_seq_len=512,
    n_head=8,
    n_layers=2,
    dropout=0.1,
)

# Compute related
ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
DEVICES = [0] if ACCELERATOR == "gpu" else 0
PRECISION = "16-mixed"
