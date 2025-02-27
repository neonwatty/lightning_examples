import os
import json
import torch
from dataclasses import dataclass
import time
from network_transformer_encoder_decoder.callbacks import MyPrintingCallback, EarlyStopping, ModelCheckpoint


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    max_seq_len: int
    n_head: int
    n_layers: int
    dropout: float


@dataclass
class DataConfig:
    num_workers: int
    vocab_size: int
    dataset_name: str
    dataset_subset: str
    source_lang: str
    target_lang: str
    max_seq_len: int
    dataset_dir: str
    source_tokenizer_path: str
    target_tokenizer_path: str


# Training hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 2

# Compute related
ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
DEVICES = [0] if ACCELERATOR == "gpu" else 0
PRECISION = "16-mixed"

def save_config_as_json(config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        if isinstance(config, dict):
            json.dump(config, f)
        else:
            json.dump(config.__dict__, f)

# create profiler_logs directory if it does not exist
if not os.path.exists("profiler_logs"):
    os.makedirs("profiler_logs")

# generate configs
# dataset_name = "Helsinki-NLP/opus_books" or "neonwatty/opus_books-sample-50"
def generate(dataset_name: str, vocab_size: int = 32000, max_seq_len: int = 512, batch_size: int = 128, d_model: int = 512, n_head: int = 8, n_layers: int = 6):
    # create run config
    run_config = {
        "dataset_name": dataset_name,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "d_model": d_model,
        "n_head": n_head,
        "n_layers": n_layers,
    }

    # add current time in unix timestamp to cache directory
    CACHE_DIR = "cache/"
    MODEL_DIR = "run-" + str(int(time.time())) + "/"

    # create subdirectories
    CONFIG_DIR = CACHE_DIR + MODEL_DIR + "/configs/"
    TOKENIZER_DIR = CACHE_DIR + MODEL_DIR + "/tokenizers/"
    CHECKPOINT_DIR = CACHE_DIR + MODEL_DIR + "/checkpoints/"
    DATASET_DIR = CACHE_DIR + "/datasets/"

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR + MODEL_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    # define paths for tokenizer
    SRC_TOKENIZER_PATH = TOKENIZER_DIR + "src_tokenizer.json"
    TGT_TOKENIZER_PATH = TOKENIZER_DIR + "tgt_tokenizer.json"

    # define paths for configs
    MODEL_CONFIG_PATH = CONFIG_DIR + "model_config.json"
    DATA_CONFIG_PATH = CONFIG_DIR + "data_config.json"
    RUN_CONFIG_PATH = CONFIG_DIR + "run_config.json"

    # Create data configurations
    DATA_CONFIG = DataConfig(
        num_workers=os.cpu_count(),
        vocab_size=vocab_size,
        dataset_name=dataset_name,
        dataset_subset="en-es",
        source_lang="en",
        target_lang="es",
        max_seq_len=max_seq_len,
        dataset_dir=DATASET_DIR,
        source_tokenizer_path=SRC_TOKENIZER_PATH,
        target_tokenizer_path=TGT_TOKENIZER_PATH,
    )

    MODEL_CONFIG = ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        n_head=n_head,
        n_layers=n_layers,
        dropout=0.1,
    )

    # choose model config and save
    save_config_as_json(MODEL_CONFIG, MODEL_CONFIG_PATH)
    save_config_as_json(DATA_CONFIG, DATA_CONFIG_PATH)
    save_config_as_json(run_config, RUN_CONFIG_PATH)

    # generate callbacks
    callbacks = [MyPrintingCallback(),
                EarlyStopping(monitor="val_loss"),
                ModelCheckpoint(
                dirpath=CHECKPOINT_DIR,
                filename='network_transformer_encoder_decoder-{epoch:02d}-{val_loss:.2f}',
                save_top_k=-1,
                every_n_epochs=1,
                )]

    return DATA_CONFIG, MODEL_CONFIG, callbacks


