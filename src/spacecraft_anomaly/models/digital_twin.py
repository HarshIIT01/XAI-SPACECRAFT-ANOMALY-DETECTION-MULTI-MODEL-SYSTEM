"""
models/digital_twin.py — Conditional VAE that generates a spacecraft "snapshot" image
                          from a telemetry embedding + anomaly state.

The generated image visualises which subsystem is under stress (colour overlays).
Since no real paired (telemetry → spacecraft-photo) data exist, the model is trained
on simulated/CAD-derived images and synthetic telemetry using domain randomisation.

Architecture
------------
  Encoder : Conv2D  →  latent (mu, log_var)   [trains on real/synthetic images]
  Decoder : Linear  →  DeConv2D               [conditioned on telemetry z + anomaly flag]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


# ──────────────────────────────────────────────────────────────
# Residual ConvBlock helpers
# ──────────────────────────────────────────────────────────────

def conv_block(in_ch, out_ch, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )

def deconv_block(in_ch, out_ch, kernel=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ──────────────────────────────────────────────────────────────
# Image Encoder (for training on reference spacecraft images)
# ──────────────────────────────────────────────────────────────

class ImageEncoder(nn.Module):
    """Encodes a (3, H, W) image into (mu, log_var) of shape (latent_dim,)."""

    def __init__(self, latent_dim: int = 64, image_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(3,   32, stride=2),   # → H/2
            conv_block(32,  64, stride=2),   # → H/4
            conv_block(64, 128, stride=2),   # → H/8
            conv_block(128, 256, stride=2),  # → H/16
        )
        flat = 256 * (image_size // 16) ** 2
        self.mu_fc   = nn.Linear(flat, latent_dim)
        self.logv_fc = nn.Linear(flat, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x).flatten(1)
        return self.mu_fc(h), self.logv_fc(h)


# ──────────────────────────────────────────────────────────────
# Conditional Image Decoder
# ──────────────────────────────────────────────────────────────

class ConditionalImageDecoder(nn.Module):
    """
    Decodes:
      z_img     — image latent (from VAE bottleneck)
      z_tel     — telemetry embedding
      anom_flag — anomaly severity [0,1]
    into a (3, H, W) spacecraft image with colour overlays.
    """

    def __init__(
        self,
        img_latent_dim: int = 64,
        tel_latent_dim: int = 32,
        image_size: int = 64,
        base_ch: int = 256,
    ):
        super().__init__()
        self.image_size = image_size
        cond_dim = img_latent_dim + tel_latent_dim + 1   # +1 for anom_flag
        init_size = image_size // 16

        self.fc = nn.Sequential(
            nn.Linear(cond_dim, base_ch * init_size * init_size),
            nn.ReLU(),
        )
        self.init_size = init_size
        self.base_ch   = base_ch

        self.deconv = nn.Sequential(
            deconv_block(base_ch,     base_ch // 2),   # → init*2
            deconv_block(base_ch // 2, base_ch // 4),  # → init*4
            deconv_block(base_ch // 4, base_ch // 8),  # → init*8
            deconv_block(base_ch // 8, 64),            # → init*16 = H
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        z_img: torch.Tensor,
        z_tel: torch.Tensor,
        anom_flag: torch.Tensor,
    ) -> torch.Tensor:
        cond = torch.cat([z_img, z_tel, anom_flag.unsqueeze(-1)], dim=-1)
        h = self.fc(cond)
        h = h.view(-1, self.base_ch, self.init_size, self.init_size)
        img = self.deconv(h)    # (B, 3, H, W)  values in [-1, 1]
        return img


# ──────────────────────────────────────────────────────────────
# Anomaly Overlay Generator
# ──────────────────────────────────────────────────────────────

class AnomalyOverlay(nn.Module):
    """
    Produces a per-pixel heatmap (red overlay) that highlights the
    subsystem most likely responsible for the anomaly.

    Inputs : telemetry embedding z_tel, anomaly severity s ∈ [0,1],
             subsystem index (0–3) as one-hot
    Output : heatmap (1, H, W) ∈ [0, 1]
    """

    N_SUBSYSTEMS = 4   # e.g. Solar Panels, Battery, Thermal, ADCS

    def __init__(self, tel_dim: int = 32, image_size: int = 64):
        super().__init__()
        self.image_size = image_size
        self.subsystem_fc = nn.Linear(tel_dim + self.N_SUBSYSTEMS, 128)
        self.deconv = nn.Sequential(
            deconv_block(128, 64),
            deconv_block(64,  32),
            deconv_block(32,  16),
            deconv_block(16,   8),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid(),
        )
        self.spatial_start = image_size // 16

    def forward(
        self,
        z_tel: torch.Tensor,             # (B, tel_dim)
        severity: torch.Tensor,          # (B,)
        subsystem_onehot: torch.Tensor,  # (B, N_SUBSYSTEMS)
    ) -> torch.Tensor:
        cond = torch.cat([z_tel, subsystem_onehot], dim=-1)
        h = F.relu(self.subsystem_fc(cond))               # (B, 128)
        h = h.view(-1, 128, 1, 1).expand(
            -1, -1, self.spatial_start, self.spatial_start
        )
        heatmap = self.deconv(h)                          # (B, 1, H, W)
        heatmap = heatmap * severity.view(-1, 1, 1, 1)
        return heatmap


# ──────────────────────────────────────────────────────────────
# Full Digital Twin
# ──────────────────────────────────────────────────────────────

class DigitalTwin(nn.Module):
    """
    End-to-end Digital Twin:
      1. Encode a reference spacecraft image (or random noise for generation-only)
      2. Decode conditioned on telemetry embedding + anomaly severity
      3. Add anomaly overlay (red heatmap on affected subsystem)

    Training:
      - Use simulated spacecraft images (CAD renders, generic satellite art)
      - Pair with synthetic telemetry (normal → no overlay, anomalous → coloured)
      - Loss = reconstruction + KL + overlay consistency
    """

    def __init__(
        self,
        img_latent_dim: int = 64,
        tel_latent_dim: int = 32,
        image_size: int = 64,
        beta: float = 1.0,
    ):
        super().__init__()
        self.img_encoder = ImageEncoder(img_latent_dim, image_size)
        self.decoder = ConditionalImageDecoder(
            img_latent_dim, tel_latent_dim, image_size
        )
        self.overlay = AnomalyOverlay(tel_latent_dim, image_size)
        self.beta = beta
        self.img_latent_dim = img_latent_dim

    def reparameterize(self, mu, log_var):
        if self.training:
            return mu + (0.5 * log_var).exp() * torch.randn_like(mu)
        return mu

    def forward(
        self,
        img: torch.Tensor,               # (B, 3, H, W) reference image
        z_tel: torch.Tensor,             # (B, tel_latent_dim)
        anom_severity: torch.Tensor,     # (B,)  ∈ [0, 1]
        subsystem_onehot: torch.Tensor,  # (B, N_SUBSYSTEMS)
    ) -> dict:
        mu, log_var = self.img_encoder(img)
        z_img = self.reparameterize(mu, log_var)

        img_hat = self.decoder(z_img, z_tel, anom_severity)        # (B, 3, H, W)
        heatmap = self.overlay(z_tel, anom_severity, subsystem_onehot)  # (B, 1, H, W)

        # Blend: overlay red channel where heatmap is high
        blended = img_hat.clone()
        blended[:, 0, :, :] = torch.clamp(
            img_hat[:, 0, :, :] + heatmap.squeeze(1), -1, 1
        )

        return {
            "img_hat"  : img_hat,
            "blended"  : blended,
            "heatmap"  : heatmap,
            "mu"       : mu,
            "log_var"  : log_var,
        }

    def generate(
        self,
        z_tel: torch.Tensor,
        anom_severity: torch.Tensor,
        subsystem_onehot: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate a twin image purely from telemetry (no reference image)."""
        if device is None:
            device = z_tel.device
        B = z_tel.size(0)
        z_img = torch.randn(B, self.img_latent_dim, device=device)
        img_hat = self.decoder(z_img, z_tel, anom_severity)
        heatmap = self.overlay(z_tel, anom_severity, subsystem_onehot)
        blended = img_hat.clone()
        blended[:, 0] = torch.clamp(img_hat[:, 0] + heatmap.squeeze(1), -1, 1)
        return blended   # (B, 3, H, W)

    def loss(
        self,
        img: torch.Tensor,
        z_tel: torch.Tensor,
        anom_severity: torch.Tensor,
        subsystem_onehot: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        out = self.forward(img, z_tel, anom_severity, subsystem_onehot)
        recon = F.mse_loss(out["img_hat"], img)
        kl    = -0.5 * torch.mean(
            1 + out["log_var"] - out["mu"].pow(2) - out["log_var"].exp()
        )
        total = recon + self.beta * kl
        return total, {"recon": recon.item(), "kl": kl.item()}


# ──────────────────────────────────────────────────────────────
# Synthetic training image generator (no real images needed)
# ──────────────────────────────────────────────────────────────

def generate_synthetic_spacecraft_image(
    image_size: int = 64,
    anomalous: bool = False,
    subsystem: int = 0,
    severity: float = 0.5,
) -> np.ndarray:
    """
    Generates a very simple synthetic spacecraft image (numpy array, HWC, uint8):
      - Grey background
      - White rectangular body
      - Blue solar panels on sides
      - If anomalous: red blob on the affected subsystem

    Subsystem indices: 0=solar_panel_L, 1=solar_panel_R, 2=main_body, 3=antenna
    """
    H = W = image_size
    img = np.full((H, W, 3), 30, dtype=np.uint8)   # dark background

    # Main body
    cy, cx = H // 2, W // 2
    bh, bw = H // 4, W // 6
    img[cy - bh:cy + bh, cx - bw:cx + bw] = [180, 180, 180]

    # Solar panels
    ph, pw = H // 8, W // 3
    img[cy - ph:cy + ph, :pw] = [30, 60, 150]        # left panel
    img[cy - ph:cy + ph, W - pw:] = [30, 60, 150]    # right panel

    # Antenna
    img[cy - bh - 8:cy - bh, cx - 2:cx + 2] = [200, 200, 50]

    if anomalous:
        sub_regions = [
            (slice(cy - ph, cy + ph), slice(0, pw)),              # L panel
            (slice(cy - ph, cy + ph), slice(W - pw, W)),          # R panel
            (slice(cy - bh, cy + bh), slice(cx - bw, cx + bw)),   # body
            (slice(cy - bh - 10, cy - bh), slice(cx - 4, cx + 4)),# antenna
        ]
        rs, cs = sub_regions[subsystem % len(sub_regions)]
        alpha = int(severity * 200)
        img[rs, cs, 0] = np.clip(img[rs, cs, 0].astype(int) + alpha, 0, 255)
        img[rs, cs, 1] = np.clip(img[rs, cs, 1].astype(int) - alpha // 2, 0, 255)
        img[rs, cs, 2] = np.clip(img[rs, cs, 2].astype(int) - alpha // 2, 0, 255)

    return img
