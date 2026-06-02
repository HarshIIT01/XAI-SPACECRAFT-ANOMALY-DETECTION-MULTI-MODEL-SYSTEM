"""
models/transformer_ad.py — Transformer-based anomaly detector.

Inspired by TranAD (Tuli et al., 2022):
  - Two-stage attention encoder
  - Focus score weighting (anomalies get amplified in stage 2)
  - Reconstruction-based scoring

Reference: https://arxiv.org/abs/2201.07284
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ──────────────────────────────────────────────────────────────
# Positional Encoding
# ──────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ──────────────────────────────────────────────────────────────
# Transformer Encoder Block
# ──────────────────────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        attn_out, attn_weights = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        if return_attn:
            return x, attn_weights
        return x


# ──────────────────────────────────────────────────────────────
# TranAD-style model
# ──────────────────────────────────────────────────────────────

class TransformerAD(nn.Module):
    """
    Two-stage transformer anomaly detector.

    Stage 1: Encode the window → latent sequence → decode
    Stage 2: Encode again, but weight input by focus scores from stage 1
             (amplifies anomalous regions → larger reconstruction error)

    Anomaly score = (1-w)*loss1 + w*loss2  where w is annealed during training.
    """

    def __init__(
        self,
        input_dim: int = 25,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        seq_len: int = 128,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # projection to model dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)

        # Stage-1 encoder + decoder
        self.enc1 = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.dec1 = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Stage-2 encoder + decoder
        self.enc2 = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.dec2 = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.out_proj = nn.Linear(d_model, input_dim)

        # focal weight (annealed 0→1 over training)
        self.register_buffer("w", torch.tensor(0.0))

    # ----------------------------------------------------------
    def _encode_decode(
        self,
        enc_layers: nn.ModuleList,
        dec_layers: nn.ModuleList,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run x through encoder layers, then decoder layers."""
        h = x
        last_attn = None
        for layer in enc_layers:
            h, attn = layer(h, return_attn=True)
            last_attn = attn
        for layer in dec_layers:
            h = layer(h)
        return h, last_attn

    # ----------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, T, C)
        Returns: x_hat1, x_hat2, attn_weights
        """
        B, T, C = x.shape
        h = self.pos_enc(self.input_proj(x))          # (B, T, D)

        # Stage 1
        h1, attn = self._encode_decode(self.enc1, self.dec1, h)
        x_hat1   = self.out_proj(h1)                  # (B, T, C)

        # Focus scores: per-time-step reconstruction error from stage 1
        focus = ((x - x_hat1) ** 2).mean(-1, keepdim=True)  # (B, T, 1)
        focus = focus / (focus.max(dim=1, keepdim=True).values + 1e-8)

        # Stage 2: weighted input
        h2 = self.pos_enc(self.input_proj(x * (1 + focus)))
        h2, _ = self._encode_decode(self.enc2, self.dec2, h2)
        x_hat2 = self.out_proj(h2)

        return x_hat1, x_hat2, attn

    # ----------------------------------------------------------
    def loss(self, x: torch.Tensor) -> torch.Tensor:
        x_hat1, x_hat2, _ = self.forward(x)
        l1 = F.mse_loss(x_hat1, x)
        l2 = F.mse_loss(x_hat2, x)
        return (1 - self.w) * l1 + self.w * l2

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-window anomaly score. Shape: (B,)"""
        with torch.no_grad():
            x_hat1, x_hat2, attn = self.forward(x)
            s1 = ((x - x_hat1) ** 2).mean(dim=(1, 2))
            s2 = ((x - x_hat2) ** 2).mean(dim=(1, 2))
            return (1 - self.w) * s1 + self.w * s2

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return stage-1 attention weights. Shape: (B, heads, T, T)"""
        with torch.no_grad():
            h = self.pos_enc(self.input_proj(x))
            _, attn = self._encode_decode(self.enc1, self.dec1, h)
            return attn

    def anneal_weight(self, epoch: int, total_epochs: int):
        """Linearly anneal the focus weight from 0 to 1."""
        self.w = torch.tensor(min(epoch / max(total_epochs - 1, 1), 1.0))


# ──────────────────────────────────────────────────────────────
# Lightweight patch-based variant for faster inference
# ──────────────────────────────────────────────────────────────

class PatchTransformerAD(nn.Module):
    """
    Splits the window into patches (like ViT for time series),
    which greatly reduces sequence length for the attention layers.
    Suitable for on-board / low-latency inference.
    """

    def __init__(
        self,
        input_dim: int = 25,
        patch_len: int = 16,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.patch_proj = nn.Linear(input_dim * patch_len, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=512, dropout=dropout)
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        self.decoder = nn.Linear(d_model, input_dim * patch_len)

    def _patchify(self, x: torch.Tensor):
        """x: (B, T, C)  →  (B, T//P, P*C)"""
        B, T, C = x.shape
        P = self.patch_len
        x = x[:, :T - T % P, :]       # trim to multiple of P
        x = x.reshape(B, -1, P * C)   # (B, n_patches, P*C)
        return x

    def forward(self, x: torch.Tensor):
        patches = self._patchify(x)                     # (B, N, P*C)
        h = self.pos_enc(self.patch_proj(patches))      # (B, N, D)
        for layer in self.encoder:
            h = layer(h)
        out = self.decoder(h)                           # (B, N, P*C)
        # reshape back to (B, N*P, C)
        B, N, PC = out.shape
        C = PC // self.patch_len
        x_hat = out.reshape(B, N * self.patch_len, C)
        return x_hat

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            T = x.size(1)
            x_hat = self.forward(x)
            T_hat = x_hat.size(1)
            x_trim = x[:, :T_hat, :]
            return ((x_trim - x_hat) ** 2).mean(dim=(1, 2))
