import torch
import pytorch_lightning as pl
from blocks import FullyConnectedBlock
from model import NN
from dataset import MnistDataModule
from callbacks import MyPrintingCallback, EarlyStopping
import config
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler

torch.set_float32_matmul_precision("medium") # to make lightning happy


if __name__ == "__main__":
    # set profiler
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("profiler_logs/profiler0"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    )

    # setup logger
    logger = TensorBoardLogger("tb_logs", name="mnist_model_v1")

    # Initialize network
    fully_connected_block = FullyConnectedBlock(input_size=config.INPUT_SIZE, num_classes=config.NUM_CLASSES)
    model = NN(fully_connected_block=fully_connected_block, num_classes=config.NUM_CLASSES)

    # initialize data module
    dm = MnistDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    # initialize trainer
    trainer = pl.Trainer(
        profiler=profiler,
        accelerator=config.ACCELERATOR,
        # devices=config.DEVICES, # only use for gpu
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
        callbacks=[MyPrintingCallback(), EarlyStopping(monitor="val_loss")],
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
