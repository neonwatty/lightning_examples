import torch
import pytest
from torch.utils.data import Dataset, DataLoader
from network_vanilla.blocks import FullyConnectedBlock
from network_vanilla.model import init_model
from network_vanilla.dataset import DataModule
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import network_vanilla.config as network_config


class OverfitCallback(Callback):
    def __init__(self, threshold=0.95):
        super().__init__()
        self.threshold = threshold

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        # Assuming the model returns 'train_accuracy' in the outputs dictionary
        current_accuracy = outputs.get('train_accuracy', 0)  # Default to 0 if key is not present

        # Check if the accuracy exceeds the threshold
        if current_accuracy >= self.threshold:
            trainer.should_stop = True
            print(f"Accuracy reached {current_accuracy:.4f}, training stopped.")


@pytest.fixture
def config():
    return {
        "sample_size": 100,
        "batch_size": 2,
        "num_batches": 10,
        "dev_mode": True,
        "data_dir": "./tests/network_vanilla/dataset",
        "num_workers": 2,
        "python_test_dataset_name": network_config.PYTHON_TEST_DATASET_NAME,
    }


@pytest.fixture
def sample_data(config):
    data_dir = config['data_dir']

    # # remove data_dir and recreate it
    # shutil.rmtree(data_dir, ignore_errors=True)
    # os.makedirs(data_dir, exist_ok=True)

    # create sample data
    data_module_sample = DataModule(
        dataset_name=config['python_test_dataset_name'],
        batch_size=config['batch_size'],
        cache_dir=config['data_dir'],
        num_workers=config['num_workers'],
    )
    data_module_sample.prepare_data()
    data_module_sample.setup()
    return data_module_sample


@pytest.fixture
def dataset(sample_data):
    return sample_data.train_dataset


@pytest.fixture
def blocks(sample_data):
    shapes = sample_data.shapes
    print(f'INFO: dataset single point shape: {shapes}')
    num_classes = shapes['num_classes']
    input_size = shapes['input_size']
    blocks = FullyConnectedBlock(input_size, num_classes)
    return blocks


@pytest.fixture
def model(sample_data):
    # unpack shared data
    num_classes = sample_data.shapes['num_classes']
    input_size = sample_data.shapes['input_size']

    # return initialized model
    return init_model(input_size, num_classes)

@pytest.fixture
def overfit_callback():
    return OverfitCallback()