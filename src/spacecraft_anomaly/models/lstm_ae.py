"""
models/lstm_ae.py — LSTM Autoencoder for unsupervised anomaly detection.

Architecture
------------
Encoder : stacked bidirectional LSTM  →  latent vector z
Decoder : stacked LSTM                →  reconstructed sequence

Anomaly score = mean squared reconstruction error per window.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.num_dirs    = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_dim * self.num_dirs, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, C)
        out, (h, c) = self.lstm(x)                     # out: (B, T, H*dirs)
        # use final hidden state from all directions
        h_final = out[:, -1, :]                         # (B, H*dirs)
        z = self.norm(self.fc(h_final))                 # (B, latent_dim)
        return z, out


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_len: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim)
        h0 = self.fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, T, H)
        out, _ = self.lstm(h0)                                     # (B, T, H)
        return self.out_proj(out)                                  # (B, T, C)


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM-AE.  Call .anomaly_score(x) to get per-sample MSE.
    """

    def __init__(
        self,
        input_dim: int = 25,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        seq_len: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = LSTMEncoder(
            input_dim, hidden_dim, latent_dim, num_layers, dropout, bidirectional=True
        )
        self.decoder = LSTMDecoder(
            latent_dim, hidden_dim, input_dim, seq_len, num_layers, dropout
        )
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, enc_out = self.encoder(x)   # (B, latent_dim)
        x_hat = self.decoder(z)        # (B, T, C)
        return x_hat, z

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-window mean squared reconstruction error. Shape: (B,)"""
        with torch.no_grad():
            x_hat, _ = self.forward(x)
            return ((x - x_hat) ** 2).mean(dim=(1, 2))

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        x_hat, _ = self.forward(x)
        return nn.functional.mse_loss(x_hat, x)


# ──────────────────────────────────────────────────────────────
# Variational variant (VAE-LSTM)
# ──────────────────────────────────────────────────────────────

class LSTMVAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=True)
        dim = hidden_dim * 2
        self.mu_fc  = nn.Linear(dim, latent_dim)
        self.log_fc = nn.Linear(dim, latent_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.mu_fc(h), self.log_fc(h)


class LSTMVariationalAE(nn.Module):
    """VAE-LSTM: adds a KL term to encourage a structured latent space."""

    def __init__(
        self,
        input_dim: int = 25,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        seq_len: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        beta: float = 1.0,   # weight of KL term
    ):
        super().__init__()
        self.encoder = LSTMVAEEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers, dropout)
        self.beta = beta
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        if self.training:
            std = (0.5 * log_var).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    def loss(self, x):
        x_hat, mu, log_var = self.forward(x)
        recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
        kl    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon + self.beta * kl, recon, kl

    def anomaly_score(self, x):
        with torch.no_grad():
            x_hat, mu, log_var = self.forward(x)
            recon_err = ((x - x_hat) ** 2).mean(dim=(1, 2))
            # also include KL term as anomaly signal
            kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1)
            return recon_err + 0.1 * kl
