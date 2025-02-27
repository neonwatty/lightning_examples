import torch
from torch import nn, optim
import torchmetrics
import pytorch_lightning as pl
from network_transformer_encoder_decoder.blocks import Transformer
from network_transformer_encoder_decoder.config import ModelDimensions
from network_transformer_encoder_decoder.config import LEARNING_RATE


class NN(pl.LightningModule):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.model = Transformer(dims)
        self.loss_fn = nn.CrossEntropyLoss()

        # accuracy options
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=dims.vocab_size)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=dims.vocab_size)

    def forward(self, x, y):
        x = self.model(x, y)
        return x

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict({"train_loss": loss, "train_accuracy": accuracy, "train_f1_score": f1_score}, on_step=True, on_epoch=True, prog_bar=True)
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

        # Flatten input and output
        x = x.reshape(x.size(0), -1)
        y = y.reshape(y.size(0), -1)

        # print shape of x and y
        scores = self.forward(x, y)

        # reshape scores for loss calculation
        scores = scores.view(-1, scores.size(-1))
        y = y.view(-1)

        # calculate loss
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


def init_model(ModelDimensions):
    block = Transformer(dims=ModelDimensions)
    return NN(block, ModelDimensions)
