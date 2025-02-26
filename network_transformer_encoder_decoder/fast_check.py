import pytorch_lightning as pl
from network_transformer_encoder_decoder.model import NN
from network_transformer_encoder_decoder.blocks import Transformer
from network_transformer_encoder_decoder.config import DATA_CONFIG_TEST, MODEL_CONFIG_TEST
from network_transformer_encoder_decoder.dataset import DataModule


if __name__ == "__main__":
    # Create a model
    model = NN(Transformer, MODEL_CONFIG_TEST)

    # Create dataset
    data = DataModule(DATA_CONFIG_TEST)

    # instantiate trainer
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, data)
    assert trainer.global_step > 0, "Trainer did not execute any steps"
