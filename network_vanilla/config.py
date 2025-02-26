import torch

# Training hyperparameters
INPUT_SIZE = 784
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 2

# Dataset
CACHE_DIR = "cache/"
NUM_WORKERS = 0
DATASET_NAME = "ylecun/mnist"
PYTHON_TEST_DATASET_NAME = "neonwatty/mnist-sample-50"

# Compute related
ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
DEVICES = 1
PRECISION = "16-mixed"
