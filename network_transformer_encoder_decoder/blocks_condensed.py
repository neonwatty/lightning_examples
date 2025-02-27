import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
from network_transformer_encoder_decoder.config import ModelConfig


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings
        return self.embedding(x) * math.sqrt(self.d_model)

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int) -> None:
        super().__init__()
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)

        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model / 2)

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(position * (10000 ** (2i / d_model))

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(position * (10000 ** (2i / d_model))

        # Convert to Tensor
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # Register the positional encoding as a buffer directly
        self.register_buffer("pe", pe)  # (1, seq_len, d_model)

    def forward(self, x):
        # Return positional encoding for the given sequence length
        return self.pe[:, : x.size(1)]


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, d_model: int, n_head: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        # token and position embeddings
        self.token_embedding = InputEmbeddings(d_model, vocab_size)
        self.positional_embedding = PositionalEncoding(d_model, seq_len)

        self.ln = LayerNorm(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model * 4, dropout=dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers, norm=self.ln)

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_embedding(x)
        x += self.positional_embedding(x)
        return self.transformer_encoder(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, d_model: int, n_head: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        # token and position embeddings
        self.token_embedding = InputEmbeddings(d_model, vocab_size)
        self.positional_embedding = PositionalEncoding(d_model, seq_len)

        self.ln = LayerNorm(d_model)
        decoder_layers = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model * 4, dropout=dropout, activation='gelu')
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=n_layers, norm=self.ln)

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        x = self.token_embedding(x)
        x += self.positional_embedding(x)
        x = self.transformer_decoder(x, memory)
        logits = F.linear(x, self.token_embedding.embedding.weight)
        return logits


class Transformer(nn.Module):
    def __init__(self, dims: ModelConfig):
        super().__init__()
        self.dims = dims
        self.encoder = Encoder(dims.vocab_size, dims.max_seq_len, dims.d_model, dims.n_head, dims.n_layers)
        self.decoder = Decoder(dims.vocab_size, dims.max_seq_len, dims.d_model, dims.n_head, dims.n_layers)

    def forward(self, x: Tensor, y: Tensor):
        x = self.encoder(x)
        x = self.decoder(y, x)
        return x
