import torch
from torch import nn, optim
import torchmetrics
import pytorch_lightning as pl
from network_vanilla.blocks import FullyConnectedBlock
from network_vanilla.config import LEARNING_RATE


class NN(pl.LightningModule):
    def __init__(self, fully_connected_block: FullyConnectedBlock, num_classes: int):
        super().__init__()
        self.fully_connected_block = fully_connected_block
        self.loss_fn = nn.CrossEntropyLoss()

        # accuracy options
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.fully_connected_block(x)
        return x

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict({"train_loss": loss, "train_accuracy": accuracy, "train_f1_score": f1_score}, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "scores": scores, "y": y, "train_accuracy": accuracy}

    # def on_train_epoch_end(self, outputs):
    #     # named method for compute at the end of an epoch only
    #     pass

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=LEARNING_RATE)


def init_model(input_size, num_classes):
    block = FullyConnectedBlock(input_size, num_classes)
    return NN(block, num_classes)
