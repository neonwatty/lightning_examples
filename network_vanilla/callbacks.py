from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from network_vanilla.config import CACHE_DIR

class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Starting to train!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")


callbacks = [MyPrintingCallback(),
             EarlyStopping(monitor="val_loss"),
             ModelCheckpoint(
             dirpath=CACHE_DIR + '/checkpoints',
             filename='network-vanilla-{epoch:02d}',
             save_top_k=-1,
             every_n_epochs=1,
            )]
