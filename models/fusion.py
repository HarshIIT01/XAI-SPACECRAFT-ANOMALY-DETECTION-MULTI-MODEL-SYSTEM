"""
models/fusion.py — Multi-modal fusion of telemetry and image embeddings.

Architecture
------------
  TelemetryBranch : GNN-VAE encoder  →  z_tel  (latent_dim,)
  ImageBranch     : CNN (ResNet-lite) →  z_img  (img_latent_dim,)
  FusionLayer     : Cross-attention or concatenation  →  z_fused
  AnomalyHead     : z_fused  →  anomaly score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ──────────────────────────────────────────────────────────────
# Lightweight CNN image encoder (no pretrained weights needed)
# ──────────────────────────────────────────────────────────────

class LightCNNEncoder(nn.Module):
    """
    A small ResNet-style CNN for satellite imagery.
    Input : (B, 3, H, W)
    Output: (B, out_dim)
    """

    class ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.GELU(),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
            )
            self.act = nn.GELU()

        def forward(self, x):
            return self.act(x + self.net(x))

    def __init__(self, out_dim: int = 64, image_size: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            self.ResBlock(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            self.ResBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(128, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.blocks(h)
        h = self.pool(h).flatten(1)
        return self.norm(self.fc(h))


# ──────────────────────────────────────────────────────────────
# Cross-Attention Fusion
# ──────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Fuses z_tel (query) with z_img (key/value) via multi-head cross-attention.
    Both are treated as single-token sequences.
    """

    def __init__(self, tel_dim: int, img_dim: int, out_dim: int, num_heads: int = 4):
        super().__init__()
        # project to common dim
        self.q_proj = nn.Linear(tel_dim, out_dim)
        self.k_proj = nn.Linear(img_dim, out_dim)
        self.v_proj = nn.Linear(img_dim, out_dim)
        self.attn   = nn.MultiheadAttention(out_dim, num_heads, batch_first=True)
        self.norm   = nn.LayerNorm(out_dim)
        self.ff     = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, z_tel: torch.Tensor, z_img: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(z_tel).unsqueeze(1)    # (B, 1, D)
        k = self.k_proj(z_img).unsqueeze(1)    # (B, 1, D)
        v = self.v_proj(z_img).unsqueeze(1)    # (B, 1, D)
        attn_out, _ = self.attn(q, k, v)
        fused = self.norm(q + attn_out).squeeze(1)   # (B, D)
        return self.norm(fused + self.ff(fused))


# ──────────────────────────────────────────────────────────────
# Concat Fusion (simpler baseline)
# ──────────────────────────────────────────────────────────────

class ConcatFusion(nn.Module):
    def __init__(self, tel_dim: int, img_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(tel_dim + img_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, z_tel: torch.Tensor, z_img: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([z_tel, z_img], dim=-1))


# ──────────────────────────────────────────────────────────────
# Anomaly Head
# ──────────────────────────────────────────────────────────────

class AnomalyHead(nn.Module):
    """
    Maps fused embedding → (anomaly_score, feature_weights).
    Dual output: reconstruction score (unsupervised) + classification logit.
    """

    def __init__(self, in_dim: int, n_channels: int, seq_len: int):
        super().__init__()
        self.reconstruct = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.GELU(),
            nn.Linear(in_dim * 2, n_channels * seq_len),
        )
        self.classify = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.n_channels = n_channels
        self.seq_len    = seq_len

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_hat = self.reconstruct(z).view(-1, self.seq_len, self.n_channels)
        logit = self.classify(z).squeeze(-1)
        return x_hat, logit


# ──────────────────────────────────────────────────────────────
# Full Multi-Modal Model
# ──────────────────────────────────────────────────────────────

class MultiModalAD(nn.Module):
    """
    Fuses telemetry embeddings with image embeddings for joint anomaly detection.

    Modes
    -----
    fusion_type = "cross_attention"  (default, stronger)
    fusion_type = "concat"           (simpler, faster)
    fusion_type = "telemetry_only"   (ablation — ignores images)
    """

    def __init__(
        self,
        n_channels: int = 25,
        tel_latent_dim: int = 32,
        img_out_dim: int = 64,
        fused_dim: int = 64,
        seq_len: int = 128,
        image_size: int = 64,
        fusion_type: str = "cross_attention",
        num_heads: int = 4,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.use_image   = fusion_type != "telemetry_only"

        # Telemetry encoder (simple biGRU for speed in fusion mode)
        self.tel_encoder = nn.Sequential(
            nn.Linear(n_channels, tel_latent_dim),
            nn.GELU(),
        )
        self.tel_gru = nn.GRU(
            tel_latent_dim, tel_latent_dim, 2,
            batch_first=True, bidirectional=True
        )
        self.tel_proj = nn.Linear(tel_latent_dim * 2, tel_latent_dim)

        # Image encoder
        if self.use_image:
            self.img_encoder = LightCNNEncoder(img_out_dim, image_size)

        # Fusion layer
        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(tel_latent_dim, img_out_dim, fused_dim, num_heads)
        elif fusion_type == "concat":
            self.fusion = ConcatFusion(tel_latent_dim, img_out_dim, fused_dim)
        else:  # telemetry only
            self.fusion = nn.Linear(tel_latent_dim, fused_dim)

        # Anomaly head
        self.head = AnomalyHead(fused_dim, n_channels, seq_len)

    def encode_telemetry(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C)  →  z_tel: (B, tel_latent_dim)"""
        h = self.tel_encoder(x)                # (B, T, D)
        out, _ = self.tel_gru(h)               # (B, T, 2D)
        return self.tel_proj(out[:, -1, :])    # (B, D)

    def forward(
        self,
        x: torch.Tensor,                        # (B, T, C) telemetry
        img: Optional[torch.Tensor] = None,     # (B, 3, H, W) image
    ) -> dict:
        z_tel = self.encode_telemetry(x)

        if self.use_image and img is not None:
            z_img  = self.img_encoder(img)
            z_fused = self.fusion(z_tel, z_img)
        elif self.use_image and img is None:
            # image branch missing: use zeros
            B = x.size(0)
            z_img   = torch.zeros(B, self.img_encoder.fc.out_features, device=x.device)
            z_fused = self.fusion(z_tel, z_img)
        else:
            z_fused = self.fusion(z_tel)

        x_hat, logit = self.head(z_fused)

        return {
            "z_tel"  : z_tel,
            "z_fused": z_fused,
            "x_hat"  : x_hat,
            "logit"  : logit,
        }

    def anomaly_score(
        self,
        x: torch.Tensor,
        img: Optional[torch.Tensor] = None,
        alpha: float = 0.7,
    ) -> torch.Tensor:
        """
        Combines reconstruction error (unsupervised) and classifier logit.
        alpha = weight of reconstruction term.
        """
        with torch.no_grad():
            out = self.forward(x, img)
            recon_score = ((x - out["x_hat"]) ** 2).mean(dim=(1, 2))
            clf_score   = torch.sigmoid(out["logit"])
            return alpha * recon_score + (1 - alpha) * clf_score

    def loss(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        img: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        out = self.forward(x, img)
        recon = F.mse_loss(out["x_hat"], x)

        if labels is not None:
            bce = F.binary_cross_entropy_with_logits(
                out["logit"], labels.float()
            )
            total = 0.7 * recon + 0.3 * bce
            return total, {"recon": recon.item(), "bce": bce.item()}

        return recon, {"recon": recon.item()}
