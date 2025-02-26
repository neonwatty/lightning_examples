import os
import torch
import pytorch_lightning as pl
from model import NN
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
import network_transformer_encoder_decoder.config as config
from network_transformer_encoder_decoder.callbacks import callbacks
from network_transformer_encoder_decoder.model import NN
from network_transformer_encoder_decoder.blocks import Transformer
from network_transformer_encoder_decoder.dataset import DataModule


torch.set_float32_matmul_precision("medium")  # to make lightning happy


if __name__ == "__main__":
    # create profiler_logs directory if it does not exist
    if not os.path.exists("profiler_logs"):
        os.makedirs("profiler_logs")

    # set profiler
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs/profiler-transformer-encoder-deocder"),
        schedule=torch.profiler.schedule(skip_first=2, wait=1, warmup=1, active=5),
    )

    # setup logger
    logger = TensorBoardLogger("tb_logs", name="transformer_ed_model_v1")

    # Initialize network
    model = NN(Transformer, config.MODEL_CONFIG_TEST)

    # initialize data module
    dm = DataModule(config.DATA_CONFIG_TEST)

    # initialize trainer
    trainer = pl.Trainer(
        # profiler=profiler,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
        callbacks=callbacks,
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
