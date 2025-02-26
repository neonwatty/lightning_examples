from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from network_transformer_encoder_decoder.config import CACHE_DIR, MODEL_DIR

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
             dirpath=CACHE_DIR + MODEL_DIR + '/checkpoints',
             filename='network_transformer_encoder_decoder-{epoch:02d}-{val_loss:.2f}',
             save_top_k=-1,
             every_n_epochs=1,
            )]
