# here we overfit a batch, and use dev mode

from network_vanilla.model import init_model
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl



# test to make sure we can overfit a single batch using pytorch lightning
def test_overfit_batch(shared_data, subtests):
    # unpack shared data
    input_size = shared_data['input_size']
    num_classes = shared_data['num_classes']
    batch_size = shared_data['batch_size']
    num_batches = shared_data['num_batches']
    dev_mode = shared_data['dev_mode']

    # instantiate model
    model = init_model(input_size, num_classes)

    # create small dataset
    class YourDataset(torch.utils.data.Dataset):
        def __len__(self):
            return batch_size

        def __getitem__(self, idx):
            x = torch.randn(input_size, num_classes)
            y = torch.randint(0, num_classes, (input_size,))
            return x, y

    trainer = pl.Trainer(fast_dev_run=True, max_epochs=100)

    dataloader = torch.utils.data.DataLoader(
        YourDataset(), batch_size=min(input_size, batch_size), shuffle=False
    ) 
    
    trainer.fit(model, train_dataloaders=dataloader)
    
    assert trainer.callback_metrics["train_loss"] < 0.01, "Model failed to overfit"
