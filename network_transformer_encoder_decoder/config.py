# Training hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 1

# Dataset hyperparameters

# Dataset
DATA_DIR = "dataset/"
NUM_WORKERS = 0
DATASET_NAME = "Helsinki-NLP/opus_books"
DATASET_SUBSET = "en-es"
PYTHON_TEST_DATASET_NAME = "neonwatty/opus_books-sample-50"
PYTHON_TEST_SUBSET_NAME = "en-es"

# Compute related
ACCELERATOR = "mps"
DEVICES = 1
PRECISION = "16-mixed"
