import torch
import pytest
from torch.utils.data import Dataset, DataLoader
from network_vanilla.blocks import FullyConnectedBlock
from network_vanilla.model import init_model
from network_vanilla.dataset import DataModule
import os
import shutil


@pytest.fixture
def config():
    return {
        "sample_size": 100,
        "batch_size": 16,
        "num_batches": 10,
        "dev_mode": True,
        "data_dir": "./tests/network_vanilla/dataset",
        "num_workers": 2,
    }


@pytest.fixture
def sample_data(config):
    data_dir = config['data_dir']

    # # remove data_dir and recreate it
    # shutil.rmtree(data_dir, ignore_errors=True)
    # os.makedirs(data_dir, exist_ok=True)

    # create sample data
    data_dir = config['data_dir']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    sample_size = config['sample_size']
    data_module_sample = DataModule(data_dir, batch_size, num_workers, sample_size)
    data_module_sample.prepare_data()
    data_module_sample.setup()
    return data_module_sample


@pytest.fixture
def dataset(sample_data):
    return sample_data.train_ds


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
