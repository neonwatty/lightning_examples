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


# Training hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 1

# Dataset params
DATA_DIR = "cache/"
NUM_WORKERS = 0
DATASET_NAME = "Helsinki-NLP/opus_books"
DATASET_SUBSET = "en-es"
PYTHON_TEST_DATASET_NAME = "neonwatty/opus_books-sample-50"
PYTHON_TEST_SUBSET_NAME = "en-es"
SOURCE_LANG = "en"
TARGET_LANG = "es"

# model params
SRC_VOCAB_SIZE = 32000
TGT_VOCAB_SIZE = 32000
D_MODEL = 1024
N_HEAD = 8
N_LAYERS = 6
DROPOUT = 0.1
MAX_LENGTH = 512

# setup model configuration
MODEL_DIMS = ModelDimensions(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    d_model=D_MODEL,
    src_seq_len=MAX_LENGTH,
    tgt_seq_len=MAX_LENGTH,
    n_head=N_HEAD,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
)

# Compute related
ACCELERATOR = "mps"
DEVICES = 1
PRECISION = "16-mixed"
