import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
from network_transformer_encoder_decoder.config import ModelDimensions


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


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.model = nn.Sequential(
            Linear(d_model, d_ff),
            nn.Dropout(dropout),
            nn.GELU(),
            Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
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

        # Convert to Tensor
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # Register the positional encoding as a buffer directly
        self.register_buffer("pe", pe)  # (1, seq_len, d_model)

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


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1, cross_attention: bool = False):
        super().__init__()

        # instantiate attention block
        self.attn = MultiHeadAttentionBlock(d_model, n_head, dropout)
        self.attn_ln = LayerNorm(d_model)

        # define cross-attention block
        self.cross_attn = MultiHeadAttentionBlock(d_model, n_head, dropout) if cross_attention else None
        self.cross_attn_ln = LayerNorm(d_model) if cross_attention else None

        # define MLP block
        n_mlp = d_model * 4
        dropout = 0.1
        self.mlp = FeedForwardBlock(d_model, n_mlp, dropout)
        self.mlp_ln = LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        # residual attention block
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]

        # residual cross-attention block
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]

        # residual MLP block
        x = x + self.mlp(self.mlp_ln(x))
        return x


class Encoder(nn.Module):
    def __init__(self, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()

        # setup positional embedding
        self.positional_embedding = PositionalEncoding(n_state, n_ctx, 0.1)

        # setup residual attention blocks
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList([ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])

        # setup layer norm
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        # apply positional embedding
        x = (x + self.positional_embedding(x)).float()

        # apply residual attention blocks
        for block in self.blocks:
            x = block(x)

        # apply layer norm
        x = self.ln_post(x)
        return x


class Decoder(nn.Module):
    def __init__(self, n_vocab: int, seq_len: int, d_model: int, n_head: int, n_layer: int):
        super().__init__()
        # setup token and positional embeddings
        self.token_embedding = InputEmbeddings(d_model, n_vocab)
        self.positional_embedding = PositionalEncoding(d_model, seq_len, 0.1)

        # setup residual attention blocks
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(d_model, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(d_model)

        # setup mask
        mask = torch.empty(seq_len, seq_len).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        # apply token and positional embeddings
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        x = x.to(xa.dtype)

        # apply residual attention blocks
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
        x = self.ln(x)

        # project back to vocab (logits)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()
        return logits


class Transformer(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = Encoder(dims.src_vocab_size, dims.d_model, dims.n_head, dims.n_layers)
        self.decoder = Decoder(dims.tgt_vocab_size, dims.max_seq_len, dims.d_model, dims.n_head, dims.n_layers)

    def forward(self, x: Tensor, xa: Tensor):
        x = self.encoder(x)
        return self.decoder(x, xa)
