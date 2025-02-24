# here we overfit a batch, and use dev mode

from network_vanilla.model import init_model
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl


def test_basic(model):
    assert isinstance(model, pl.LightningModule)
    assert len(list(model.parameters())) > 0


def test_forward(sample_data, model, subtests):
    # pass single datapoint through model
    x, y = next(iter(sample_data.train_dataloader()))

    # take single datapoint
    x = x[0:1]
    y = y[0:1]

    # pass through model
    out = model.forward(x)

    # check shapes
    num_classes = sample_data.shapes['num_classes']
    with subtests.test(msg="single shape is incorrect"):
        assert out.shape == (1, num_classes), "single shape is incorrect"

    # pass batch through model
    out = model.forward(x)

    # check output shape
    with subtests.test(msg="batch shape is incorrect"):
        assert out.shape == (x.shape[0], num_classes), "batch shape is incorrect"
    with subtests.test(msg="batch dtype is incorrect"):
        assert out.dtype == torch.float32, "batch dtype is incorrect"


def test_backward(sample_data, model):
    # unpack batch
    x, y = next(iter(sample_data.train_dataloader()))

    # pass through model
    loss = model.training_step((x, y), 0)['loss']

    # check loss
    assert loss.item() > 0, "loss is zero"
    loss.backward()

    # check gradients
    for param in model.parameters():
        assert param.grad is not None, "gradient is None"
        assert torch.any(param.grad != 0), "gradient is zero"
        break
    else:
        assert False, "no parameters to check"


# test to make sure we can overfit a single batch using pytorch lightning
def test_overfit_batch(sample_data, model, overfit_callback):
    # overfit a single batch
    trainer = pl.Trainer(overfit_batches=1, max_epochs=10, callbacks=[overfit_callback])

    # fit model
    trainer.fit(model, sample_data)

    # check overfitting
    assert trainer.callback_metrics['train_accuracy'] > 0.9, "accuracy is too low"








