from dataclasses import dataclass


@dataclass
class ModelDimensions:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int
    src_seq_len: int
    tgt_seq_len: int
    n_head: int
    n_layers: int
    dropout: float


@dataclass
class DataConfig:
    data_dir: str
    num_workers: int
    dataset_name: str
    dataset_subset: str
    python_test_dataset_name: str
    python_test_subset_name: str
    source_lang: str
    target_lang: str


# Training hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 1

# Create data configuration
data_config = DataConfig(
    cache_dir="cache/",
    num_workers=0,
    dataset_name="neonwatty/opus_books-sample-50",
    dataset_subset="en-es",
    source_lang="en",
    target_lang="es",
)

# setup model configuration
MODEL_DIMS = ModelDimensions(
    src_vocab_size=32000,
    tgt_vocab_size=32000,
    d_model=1024,
    max_seq_len=512,
    n_head=8,
    n_layers=6,
    dropout=0.1,
)

# Compute related
ACCELERATOR = "mps"
DEVICES = 1
PRECISION = "16-mixed"
