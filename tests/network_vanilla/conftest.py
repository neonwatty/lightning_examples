import pytest


@pytest.fixture
def shared_data():
    return {
        'input_size': 10,
        'num_classes': 5
    }