import os
import pytorch_lightning as pl
import torch
from network_transformer_encoder_decoder.model import NN
from network_transformer_encoder_decoder.config import DataConfig, ModelDimensions
import json
from tokenizers import Tokenizer


def load():
    # Define the path to the checkpoint and configs and tokenizers
    run_dir = "./cache/run-1740610340/"
    checkpoint_dir = run_dir + "checkpoints/"
    tokenizer_dir = run_dir + "tokenizers/"
    config_dir = run_dir + "configs/"

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

    # Load the checkpoint
    checkpoint = torch.load(last_checkpoint)

    # Load the model state_dict from the checkpoint
    model.load_state_dict(checkpoint['state_dict'])

    # Set the model to evaluation mode
    model.eval();

    return src_tokenizer, tgt_tokenizer, model


def test():
    src_tokenizer, tgt_tokenizer, model = load()
    max_seq_len = 64
    test_input = "hi there"
    source_text = "[BOS]" + " " + test_input + " " + "[EOS]"
    source_tokens = src_tokenizer.encode(test_input).ids
    source_padding = max_seq_len - len(source_tokens)
    source_tokens = source_tokens[: max_seq_len]
    if source_padding > 0:
        source_tokens += [src_tokenizer.token_to_id("[PAD]")] * source_padding

    test_output = "[BOS]"
    target_tokens = tgt_tokenizer.encode(test_output).ids

    # Perform inference (translate)
    with torch.no_grad():
        for i in range(10):
            print(i)
            # Pass input through the model
            output = model(torch.tensor(source_tokens).unsqueeze(0), torch.tensor(target_tokens).unsqueeze(0))

            # Get the last token prediction (top of the sequence)
            predicted_token = torch.argmax(output[:, -1, :], dim=-1)
            print(predicted_token)

            # Append the predicted token to the target input for the next step
            target_tokens.append(predicted_token.item())
            print(target_tokens)

            # If the model generates the EOS token, break early
            if predicted_token.item() == tgt_tokenizer.token_to_id("[EOS]"):
                break

    # Decode the output token ids to text
    output_text = tgt_tokenizer.decode(target_tokens)
