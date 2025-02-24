import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def test_basic(dataset, blocks):
    assert isinstance(blocks, nn.Module)
    assert len(list(blocks.parameters())) == 4


def test_forward(dataset, dataloader, blocks, subtests):
    # pass single datapoint through blocks
    x, y = dataset[0]
    out = blocks(x)

    # unpack num_classes from dataset
    shape_dict = dataset.shapes()
    num_classes = shape_dict['num_classes']

    # check output shape
    with subtests.test(msg="single shape is incorrect"):
        assert out.shape == (num_classes,), "single shape is incorrect"
    with subtests.test(msg="single dtype is incorrect"):
        assert out.dtype == torch.float32, "single dtype is incorrect"
    
    # pass batch through blocks
    batch = next(iter(dataloader)) 
    x_batch, y_batch = batch
    out = blocks.forward(x_batch)

    # check output shape
    with subtests.test(msg="batch shape is incorrect"):
        assert out.shape == (x_batch.shape[0], num_classes), "batch shape is incorrect"
    with subtests.test(msg="batch dtype is incorrect"):
        assert out.dtype == torch.float32, "batch dtype is incorrect"


def test_backward(dataset, dataloader, blocks):
    # unpack batch
    batch = next(iter(dataloader)) 
    x, y = batch

    # pass through block
    out = blocks(x)
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
