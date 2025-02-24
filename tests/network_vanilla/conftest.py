import torch
import pytest
from torch.utils.data import Dataset, DataLoader
from network_vanilla.blocks import FullyConnectedBlock
from network_vanilla.model import init_model


class TestDataset(Dataset):
    def __init__(self, num_samples=100, input_dim=10, num_classes=5):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.x = torch.randn(num_samples, input_dim)
        self.y = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def shape(self):
        return self.x.shape, self.y.shape


@pytest.fixture
def shared_data():
    return {
        "num_samples": 100,
        "input_size": 15,
        "num_classes": 5,
        "batch_size": 16,
        "num_batches": 10,
        "dev_mode": True,
    }


@pytest.fixture
def dataset(shared_data):
    # unpack shared data
    num_samples = shared_data['num_samples']
    input_size = shared_data['input_size']
    num_classes = shared_data['num_classes']

    # return dataset
    return TestDataset(num_samples, input_size, num_classes)


@pytest.fixture
def dataloader(dataset):
    return DataLoader(dataset, batch_size=16, shuffle=True)


@pytest.fixture
def blocks(dataset):
    blocks = FullyConnectedBlock(dataset.x.shape[1], dataset.y.shape[0])
    return blocks

@pytest.fixture
def model(blocks):
    # get test dataset shape
    x_shape, y_shape = dataset.shape()
    input_size = x_shape[1]
    num_classes = y_shape[0]

    # return initialized model
    return init_model(input_size, num_classes)
