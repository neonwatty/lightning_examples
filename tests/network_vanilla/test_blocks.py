import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def test_basic(dataset, blocks):
    assert isinstance(blocks, nn.Module)
    assert len(list(blocks.parameters())) == 4


def test_forward(sample_data, blocks, subtests):
    # pass single datapoint through blocks
    x, y = next(iter(sample_data.train_dataloader()))

    # take single datapoint
    x = x[0:1]
    y = y[0:1]

    # pass through blocks
    out = blocks.forward(x)

    # check shapes
    num_classes = sample_data.shapes['num_classes']
    with subtests.test(msg="single shape is incorrect"):
        assert out.shape == (1, num_classes), "single shape is incorrect"

    # pass batch through blocks
    out = blocks.forward(x)

    # check output shape
    with subtests.test(msg="batch shape is incorrect"):
        assert out.shape == (x.shape[0], num_classes), "batch shape is incorrect"
    with subtests.test(msg="batch dtype is incorrect"):
        assert out.dtype == torch.float32, "batch dtype is incorrect"


def test_backward(sample_data, blocks):
    # unpack batch
    x, y = next(iter(sample_data.train_dataloader()))

    # pass through block
    out = blocks(x)

    # add loss
    loss = F.cross_entropy(out, y)

    # check loss
    assert loss.item() > 0, "loss is zero"
    loss.backward()

    # check gradients
    for param in blocks.parameters():
        assert param.grad is not None, "gradient is None"
        assert torch.any(param.grad != 0), "gradient is zero"
        break
    else:
        assert False, "no parameters to check"
