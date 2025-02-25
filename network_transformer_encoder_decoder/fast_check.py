import pytorch_lightning as pl
from network_transformer_encoder_decoder.model import NN
from network_transformer_encoder_decoder.blocks import Transformer
from network_transformer_encoder_decoder.config import ModelDimensions
from network_transformer_encoder_decoder.dataset import DataLoader


if __name__ == "__main__":
    # Create a model
    model = NN(Transformer(ModelDimensions()), ModelDimensions())
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, sample_data)
    assert trainer.global_step > 0, "Trainer did not execute any steps"
