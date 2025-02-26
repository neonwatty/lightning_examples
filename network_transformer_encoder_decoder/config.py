import os
import torch
from dataclasses import dataclass
import time


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
    num_workers: int
    dataset_name: str
    dataset_subset: str
    source_lang: str
    target_lang: str
    max_seq_len: int


# Training hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 2
# add current time in unix timestamp to cache directory
CACHE_DIR = "cache/"
MODEL_DIR = "run-" + str(int(time.time())) + "/"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR + MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR + MODEL_DIR + '/checkpoints', exist_ok=True)
os.makedirs(CACHE_DIR + MODEL_DIR + '/tokenizers', exist_ok=True)

# Create data configurations
DATA_CONFIG_TEST = DataConfig(
    num_workers=os.cpu_count(),
    dataset_name="neonwatty/opus_books-sample-50",
    dataset_subset="en-es",
    source_lang="en",
    target_lang="es",
    max_seq_len=128,
)

DATA_CONFIG = DataConfig(
    num_workers=os.cpu_count(),
    dataset_name="Helsinki-NLP/opus_books",
    dataset_subset="en-es",
    source_lang="en",
    target_lang="es",
    max_seq_len=128,
)


# setup model configurations
MODEL_CONFIG_TEST = ModelDimensions(
    src_vocab_size=3200,
    tgt_vocab_size=3200,
    d_model=32,
    max_seq_len=128,
    n_head=8,
    n_layers=1,
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
