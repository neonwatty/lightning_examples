# Training hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 1

# Dataset hyperparameters

# Dataset
DATA_DIR = "cache/"
NUM_WORKERS = 0
DATASET_NAME = "Helsinki-NLP/opus_books"
DATASET_SUBSET = "en-es"
PYTHON_TEST_DATASET_NAME = "neonwatty/opus_books-sample-50"
PYTHON_TEST_SUBSET_NAME = "en-es"
SOURCE_LANG = "en"
TARGET_LANG = "es"
SRC_VOCAB_SIZE = 32000
TGT_VOCAB_SIZE = 32000
D_MODEL = 1024
SRC_SEQ_LEN = 512
TGT_SEQ_LEN = 512
N_HEAD = 8
N_LAYERS = 6
DROPOUT = 0.1

# Compute related
ACCELERATOR = "mps"
DEVICES = 1
PRECISION = "16-mixed"
