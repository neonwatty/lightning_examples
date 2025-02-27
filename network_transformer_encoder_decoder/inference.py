import os
import pytorch_lightning as pl
import torch
from network_transformer_encoder_decoder.model import NN
from network_transformer_encoder_decoder.blocks_condensed import Transformer
from network_transformer_encoder_decoder.config import DataConfig, ModelDimensions
import json
from tokenizers import Tokenizer


def load(run_dir):
    # Define the path to the checkpoint and configs and tokenizers
    total_run_dir = "./cache" + "/" + run_dir + "/"
    checkpoint_dir = total_run_dir + "checkpoints/"
    tokenizer_dir = total_run_dir + "tokenizers/"
    config_dir = total_run_dir + "configs/"

    # load tokenizer
    src_tokenizer = Tokenizer.from_file(tokenizer_dir + "src_tokenizer.json")
    tgt_tokenizer = Tokenizer.from_file(tokenizer_dir + "tgt_tokenizer.json")

    # load in checkpoint file names
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    checkpoint_files.sort()
    last_checkpoint = checkpoint_dir + checkpoint_files[-1]

    # load in the last checkpoint
    with open(config_dir + "model_config.json", "r") as f:
        model_config = json.load(f)

    # convert model_config to ModelDimensions
    model_dimensions = ModelDimensions(**model_config)

    # load in config_dir + data_config.json
    with open(config_dir + "data_config.json", "r") as f:
        data_config = json.load(f)

    # convert data_config to DataConfig
    data_config = DataConfig(**data_config)

    # Create your model instance
    model = NN(model_dimensions)

    # Create transformer instance
    transformer = Transformer(model_dimensions)

    # Load the checkpoint
    checkpoint = torch.load(last_checkpoint)

    # Load the model state_dict from the checkpoint
    model.load_state_dict(checkpoint['state_dict'])

    # Set the model to evaluation mode
    model.eval();

    return src_tokenizer, tgt_tokenizer, model, transformer


import torch

def test(test_input, src_tokenizer, tgt_tokenizer, model):
    max_seq_len = 64  # Keep it fixed

    # Tokenize source input
    source_tokens = src_tokenizer.encode("[BOS] " + test_input + " [EOS]").ids
    source_tokens = source_tokens[:max_seq_len]  # Truncate if needed
    source_padding = max_seq_len - len(source_tokens)

    # Pad source tokens if necessary
    if source_padding > 0:
        source_tokens += [src_tokenizer.token_to_id("[PAD]")] * source_padding

    # Convert source tokens to tensor and move to device
    source_tensor = torch.tensor(source_tokens, dtype=torch.long).unsqueeze(0)

    # Initialize target sequence with BOS token
    target_tokens = [tgt_tokenizer.token_to_id("[BOS]")]
    target_tensor = torch.tensor(target_tokens, dtype=torch.long).unsqueeze(0)

    # Perform inference (greedy decoding)
    model.eval()
    with torch.no_grad():
        for _ in range(max_seq_len):  # Generate up to max_seq_len tokens
            output = model(source_tensor, target_tensor)
            predicted_token = torch.argmax(output[:, -1, :], dim=-1).item()

            # Append the predicted token
            target_tokens.append(predicted_token)

            # If EOS is generated, stop early
            if predicted_token == tgt_tokenizer.token_to_id("[EOS]"):
                break

            # Update target tensor
            target_tensor = torch.tensor(target_tokens, dtype=torch.long).unsqueeze(0)
            print(target_tensor)
    # Decode and return generated text
    return tgt_tokenizer.decode(target_tokens)

