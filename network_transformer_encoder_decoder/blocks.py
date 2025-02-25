import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Optional, Tuple
from contextlib import contextmanager

try:
    from torch.nn.functional import scaled_dot_product_attention

    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.model = nn.Sequential(
            Linear(d_model, d_ff),
            nn.Dropout(dropout),
            Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.model(x)


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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

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

        # Add a batch dimension to the positional encoding
        self.pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # Register the positional encoding as a buffer
        self.register_buffer("pe", self.pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)  # (batch, seq_len, d_model)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(features)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.n_head = n_head  # Number of heads

        # Make sure d_model is divisible by h
        assert d_model % n_head == 0, "d_model is not divisible by h"

        self.d_k = d_model // n_head  # Dimension of vector seen by each head
        self.w_q = Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = Linear(d_model, d_model, bias=False)  # Wv
        self.w_out = Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, xa: Optional[Tensor], mask: Optional[Tensor], kv_cache: Optional[dict] = None):
        # transform x to query
        q = self.w_q(x)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # Determine the input to use for key and value projections
        input_tensor = x if xa is None else xa

        # Check if KV cache is available and valid
        if kv_cache is None or self.key not in kv_cache:
            # If no cache, or cache is invalid, perform key/value projections
            k = self.key(input_tensor)
            v = self.value(input_tensor)
        else:
            # Otherwise, use cached key and value tensors
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        # compute attention
        wv, qk = self.qkv_attention(q, k, v, mask)

        # Apply output projection
        out = self.w_out(wv)
        return out, qk

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, seq_len, d_model = q.shape
        scale = (d_model // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if SDPA_AVAILABLE:
            a = scaled_dot_product_attention(q, k, v, is_causal=mask is not None and seq_len > 1)
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:seq_len, :seq_len]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        return out, qk
