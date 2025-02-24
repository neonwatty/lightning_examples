# here we overfit a batch, and use dev mode

from network_vanilla.model import init_model
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl


# test to make sure we can overfit a single batch using pytorch lightning
def test_overfit_batch(shared_data, dataloader, blocks):
    # unpack shared data
    dev_mode = shared_data['dev_mode']
    num_batches = shared_data['num_batches']

    # set seed
    pl.seed_everything(42)

    # init model
    model = init_model(blocks)

    # init trainer
    trainer = pl.Trainer(max_epochs=1, limit_train_batches=num_batches, limit_val_batches=num_batches, fast_dev_run=dev_mode)

    # fit model
    trainer.fit(model, dataloader)

    # test model
    trainer.test(model, dataloader)

    # check that loss decreased
    assert trainer.callback_metrics['train_loss'] < 0.1
    assert trainer.callback_metrics['val_loss'] < 0.1
    assert trainer.callback_metrics['test_loss'] < 0.1