import os
import pytorch_lightning as pl
import torch
from network_transformer_encoder_decoder.model import NN
from network_transformer_encoder_decoder.blocks_condensed import Transformer
from network_transformer_encoder_decoder.config import DataConfig, ModelConfig
from network_transformer_encoder_decoder.dataset import DataModule
import json
from tokenizers import Tokenizer


def load_model(run_dir):
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

    # convert model_config to ModelConfig
    model_config = ModelConfig(**model_config)

    # load in config_dir + data_config.json
    with open(config_dir + "data_config.json", "r") as f:
        data_config = json.load(f)

    # convert data_config to DataConfig
    data_config = DataConfig(**data_config)

    # Create your model instance
    model = NN(model_config)

    # Create transformer instance
    transformer = Transformer(model_config)

    # Load the checkpoint
    checkpoint = torch.load(last_checkpoint)

    # Load the model state_dict from the checkpoint
    model.load_state_dict(checkpoint['state_dict'])

    # Set the model to evaluation mode
    model.eval();

    return data_config, model_config, src_tokenizer, tgt_tokenizer, model, transformer


def greedy_decode(model, src_tokenizer, tgt_tokenizer, test_input, max_seq_len):
    # Tokenize source input
    source_tokens = src_tokenizer.encode("[BOS] " + test_input + " [EOS]").ids
    source_tokens = source_tokens[:max_seq_len]  # Truncate if needed
    source_padding = max_seq_len - len(source_tokens)

    # Pad source tokens if necessary
    if source_padding > 0:
        source_tokens += [src_tokenizer.token_to_id("[PAD]")] * source_padding
    
    # Convert to tensor
    source_tokens = torch.tensor(source_tokens, dtype=torch.long).unsqueeze(0)

    # precompute the encoder output and re-use for all the decoding steps
    encoder_output = model.encoder(source_tokens)

    # Initialize target sequence with BOS token
    decoder_tokens = [tgt_tokenizer.token_to_id("[BOS]")]
    decoder_output = torch.empty(1, 1, dtype=torch.long).fill_(decoder_tokens[0]).type_as(source_tokens)

    # Perform inference (greedy decoding)
    with torch.no_grad():
        while True:
            print(f'deocder_output shape: {decoder_output.shape}')
            print(f'encoder_output shape: {encoder_output.shape}')

            # stopping condition
            if decoder_output.size(1) > max_seq_len:
                break
            
            # decoder_output: (batch, seq_len)
            output = model.decoder(decoder_output, encoder_output)
            predicted_token = torch.argmax(output[:, -1, :], dim=-1).item()

            # Append the predicted token
            decoder_tokens.append(predicted_token)

            # If EOS is generated, stop early
            if predicted_token == tgt_tokenizer.token_to_id("[EOS]"):
                break

            # Update target tensor
            decoder_tokens = torch.tensor(decoder_tokens, dtype=torch.long).unsqueeze(0)

    # Decode and return generated text
    return tgt_tokenizer.decode(decoder_token)


def test(data_config, Modemodel_configlConfig, src_tokenizer, tgt_tokenizer, model):
    # Set the maximum sequence length
    max_seq_len = data_config.max_seq_len

    # # instantiate data module
    # dm = DataModule(data_config)

    # # get validation dataset
    # val_dataset = dm.val_dataloader()

    # # sample a single example from the validation dataset
    # test_input = val_dataset[0]["source_text"]
    # print(f'Input: {test_input}')
    test_input = "I am a test input"

    # Perform greedy decoding
    decoded_output = greedy_decode(model, src_tokenizer, tgt_tokenizer, test_input, max_seq_len)

    # Print the decoded output
    print(f"Input: {test_input}")
    print(f"Output: {decoded_output}")
    print()


