import pytorch_lightning as pl
from blocks import FullyConnectedBlock
from model import NN
from dataset import MnistDataModule
import config

if __name__ == "__main__":
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
        accelerator=config.ACCELERATOR,
        # devices=config.DEVICES, # only use for gpu
        min_epochs=1,
        max_epochs=3,
        precision=config.PRECISION,
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
