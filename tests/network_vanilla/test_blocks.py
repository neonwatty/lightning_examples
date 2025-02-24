from network_vanilla import blocks
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def test_basic(dataset, blocks):
    assert isinstance(blocks, nn.Module)
    assert len(list(blocks.parameters())) == 4


def test_forward(dataset, subtests):
    # pass single datapoint through blocks
    out = blocks(dataset.x[0])

    # unpack num_classes from dataset
    input_size = dataset.x.shape[1]
    num_classes = dataset.y.shape[0]

    # check output shape
    with subtests.test(msg="single shape is incorrect"):
        assert out.shape == (1, num_classes), "single shape is incorrect"
    with subtests.test(msg="single dtype is incorrect"):
        assert out.dtype == torch.float32, "single dtype is incorrect"
    
    # pass batch through blocks
    out = blocks(dataset.x)

    # check output shape
    with subtests.test(msg="batch shape is incorrect"):
        assert out.shape == (dataset.x.shape[0], num_classes), "batch shape is incorrect"
    with subtests.test(msg="batch dtype is incorrect"):
        assert out.dtype == torch.float32, "batch dtype is incorrect"


def test_backward(shared_data):
    # unpack shared data
    input_size = shared_data['input_size']
    num_classes = shared_data['num_classes']

    # instantiate block
    block = blocks.FullyConnectedBlock(input_size, num_classes)

    # create single test datapoint
    x = torch.tensor(np.random.rand(1, input_size), dtype=torch.float32)
    y = torch.tensor(np.random.randint(0, num_classes, (1,)), dtype=torch.long)

    # pass through block
    out = block(x)
    loss = F.cross_entropy(out, y)

    # check backward
    loss.backward()
    for p in block.parameters():
        assert p.grad is not None, "gradients are not computed"
        assert torch.any(p.grad != 0), "gradients are all zeros"
        p.grad.zero_()
