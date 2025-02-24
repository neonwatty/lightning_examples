import torch
import pytest
from torch.utils.data import Dataset, DataLoader
from network_vanilla.blocks import FullyConnectedBlock
from network_vanilla.model import init_model


class TestDataset(Dataset):
    def __init__(self, num_samples=100, input_size=10, num_classes=5):
        self.num_samples = num_samples
        self.input_size = input_size
        self.num_classes = num_classes
        self.x = torch.randn(num_samples, input_size)
        self.y = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def shapes(self):
        return {"num_samples": self.num_samples, "input_size": self.input_size, "num_classes": self.num_classes}


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
def dataloader(shared_data, dataset):
    batch_size = shared_data['batch_size']
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@pytest.fixture
def blocks(shared_data, dataset):
    num_classes = shared_data['num_classes']
    input_size = shared_data['input_size']
    blocks = FullyConnectedBlock(input_size, num_classes)
    return blocks


@pytest.fixture
def model(shared_data, blocks):
    # unpack shared data
    input_size = shared_data['input_size']
    num_classes = shared_data['num_classes']

    # return initialized model
    return init_model(input_size, num_classes)
