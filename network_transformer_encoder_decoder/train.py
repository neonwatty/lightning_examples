import os
import json
import torch
import pytorch_lightning as pl
from model import NN
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from network_transformer_encoder_decoder.config import generate, ACCELERATOR, DEVICES, NUM_EPOCHS, PRECISION
from network_transformer_encoder_decoder.model import NN
from network_transformer_encoder_decoder.dataset import DataModule


torch.set_float32_matmul_precision("medium")  # to make lightning happy


if __name__ == "__main__":
    # generate configs
    data_config, model_config, callbacks = generate(
        dataset_name="Helsinki-NLP/opus_books", vocab_size=32000, max_seq_len=256, batch_size=128, d_model=512, n_head=8, n_layers=6
    ) # neonwatty/opus_books-sample-50 or Helsinki-NLP/opus_books

    # set profiler
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs/profiler-transformer-encoder-deocder"),
        schedule=torch.profiler.schedule(skip_first=2, wait=1, warmup=1, active=5),
    )

    # setup logger
    logger = TensorBoardLogger("tb_logs", name="transformer_ed_model_v1")

    # Initialize network
    model = NN(model_config)

    # initialize data module
    dm = DataModule(data_config)

    # initialize trainer
    trainer = pl.Trainer(
        # profiler=profiler,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        min_epochs=1,
        max_epochs=NUM_EPOCHS,
        precision=PRECISION,
        callbacks=callbacks,
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
