"""
train.py — Unified training script for all model types.

Usage
-----
  python train.py --model GNN --dataset SMAP --channel P-1 --epochs 50
  python train.py --model TRANSFORMER --dataset OPSSAT --epochs 30
  python train.py --model LSTM_AE --dataset MSL --channel C-1
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import Config, DataConfig, ModelConfig, TrainConfig
from data.smap_msl import SMAPMSLLoader
from data.opssat   import OpsSatLoader


# ──────────────────────────────────────────────────────────────
# Device selection
# ──────────────────────────────────────────────────────────────

def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


# ──────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────

def build_model(model_type: str, n_channels: int, cfg: ModelConfig, seq_len: int):
    if model_type == "LSTM_AE":
        from models.lstm_ae import LSTMAutoencoder
        return LSTMAutoencoder(
            input_dim=n_channels,
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            seq_len=seq_len,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
    elif model_type == "LSTM_VAE":
        from models.lstm_ae import LSTMVariationalAE
        return LSTMVariationalAE(
            input_dim=n_channels,
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            seq_len=seq_len,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
    elif model_type == "TRANSFORMER":
        from models.transformer_ad import TransformerAD
        return TransformerAD(
            input_dim=n_channels,
            d_model=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            ff_dim=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            seq_len=seq_len,
        )
    elif model_type == "GNN":
        from models.graph_rnn import GNNVariationalAD
        return GNNVariationalAD(
            n_channels=n_channels,
            gnn_dim=cfg.hidden_dim // 2,
            gru_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            seq_len=seq_len,
            dropout=cfg.dropout,
        )
    elif model_type == "FUSION":
        from models.fusion import MultiModalAD
        return MultiModalAD(
            n_channels=n_channels,
            tel_latent_dim=cfg.latent_dim,
            fused_dim=cfg.hidden_dim,
            seq_len=seq_len,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ──────────────────────────────────────────────────────────────
# Training step
# ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, model_type, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    # Anneal Transformer focus weight
    if model_type == "TRANSFORMER" and hasattr(model, "anneal_weight"):
        model.anneal_weight(epoch, total_epochs)

    for x, y in loader:
        x = x.to(device)   # (B, T, C)
        optimizer.zero_grad()

        if model_type in ("LSTM_AE",):
            loss = model.reconstruction_loss(x)

        elif model_type in ("LSTM_VAE", "GNN"):
            loss, recon, kl = model.loss(x)

        elif model_type == "TRANSFORMER":
            loss = model.loss(x)

        elif model_type == "FUSION":
            labels = y.to(device) if y is not None else None
            loss, _ = model.loss(x, labels)

        else:
            loss = model.reconstruction_loss(x)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ──────────────────────────────────────────────────────────────
# Validation step
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        scores = model.anomaly_score(x).cpu().numpy()
        all_scores.append(scores)
        all_labels.append(y.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


# ──────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────

def train(
    model_type: str = "GNN",
    dataset: str = "SMAP",
    channel: str = "P-1",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    window_size: int = 128,
    device_pref: str = "auto",
    save_dir: str = "checkpoints",
):
    cfg = Config()
    device = get_device(device_pref)
    print(f"\n{'='*60}")
    print(f" Spacecraft Anomaly Detection — Training")
    print(f" Model: {model_type}  |  Dataset: {dataset}/{channel}")
    print(f" Device: {device}  |  Epochs: {epochs}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    if dataset in ("SMAP", "MSL"):
        loader = SMAPMSLLoader(
            root_dir=cfg.data.smap_msl_dir,
            channel=channel,
            spacecraft=dataset,
            window_size=window_size,
        )
        loader.summary()
        train_dl, val_dl = loader.get_loaders(batch_size=batch_size)
        n_channels = loader.n_channels
    else:
        loader = OpsSatLoader(
            channel_id=channel,
            window_size=window_size,
        )
        loader.summary()
        train_dl, val_dl = loader.get_loaders(batch_size=batch_size)
        n_channels = loader.n_channels

    # ── Model ────────────────────────────────────────────────
    model = build_model(model_type, n_channels, cfg.model, seq_len=window_size)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # ── Training loop ────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_dl, optimizer, device, model_type, epoch, epochs)
        scheduler.step()

        # Validation
        val_scores, val_labels = validate(model, val_dl, device)

        # Quick F1 at 99th percentile threshold
        from sklearn.metrics import f1_score
        from data.preprocessing import point_adjust
        thr   = np.percentile(val_scores, 99)
        preds = point_adjust(val_labels, (val_scores > thr).astype(int))
        f1    = f1_score(val_labels, preds, zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_f1"].append(f1)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{epochs}  |  loss={train_loss:.5f}  "
              f"f1={f1:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
              f"({elapsed:.1f}s)")

        # Checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            ckpt_path = os.path.join(save_dir, f"{model_type}_{dataset}_{channel}_best.pt")
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "loss"       : best_loss,
                "f1"         : f1,
                "n_channels" : n_channels,
                "window_size": window_size,
                "model_type" : model_type,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {cfg.train.patience} epochs)")
                break

    print(f"\n✅ Training complete. Best loss: {best_loss:.5f}")
    print(f"   Checkpoint saved: {ckpt_path}")
    return model, history


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train spacecraft anomaly detector")
    parser.add_argument("--model",   default="GNN",
                        choices=["LSTM_AE", "LSTM_VAE", "TRANSFORMER", "GNN", "FUSION"])
    parser.add_argument("--dataset", default="SMAP", choices=["SMAP", "MSL", "OPSSAT"])
    parser.add_argument("--channel", default="P-1")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--batch",   type=int, default=64)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--window",  type=int, default=128)
    parser.add_argument("--device",  default="auto")
    parser.add_argument("--save",    default="checkpoints")
    args = parser.parse_args()

    train(
        model_type=args.model,
        dataset=args.dataset,
        channel=args.channel,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        window_size=args.window,
        device_pref=args.device,
        save_dir=args.save,
    )