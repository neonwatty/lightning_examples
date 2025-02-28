import pytorch_lightning as pl
from network_transformer_encoder_decoder.model import NN
from network_transformer_encoder_decoder.config import generate
from network_transformer_encoder_decoder.dataset import DataModule


if __name__ == "__main__":
    # generate configs
    data_config, model_config, callbacks = generate(
        dataset_name="Helsinki-NLP/opus_books", vocab_size=320, max_seq_len=512, batch_size=128, d_model=32, n_head=2, n_layers = 2
    )

    # Create a model
    model = NN(model_config)

    # Create dataset
    data = DataModule(data_config)

    # instantiate trainer
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, data)
    assert trainer.global_step > 0, "Trainer did not execute any steps"
