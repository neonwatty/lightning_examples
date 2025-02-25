# Training hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 1

# Dataset hyperparameters

# Dataset
DATA_DIR = "dataset/"
NUM_WORKERS = 0
DATASET_NAME = "ylecun/mnist"
PYTHON_TEST_DATASET_NAME = "neonwatty/mnist-sample-50"

# Compute related
ACCELERATOR = "mps"
DEVICES = 1
PRECISION = "16-mixed"
