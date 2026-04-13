"""
data/smap_msl.py — Loader for the NASA SMAP / MSL anomaly benchmark.

Dataset layout expected (after downloading from:
  https://s3-us-west-2.amazonaws.com/telemanom/data.zip  ):

  data/raw/SMAP_MSL/
      train/   <channel>.npy   — (T_train, C) float32
      test/    <channel>.npy   — (T_test,  C) float32
  labeled_anomalies.csv        — columns: chan_id, spacecraft, anomaly_sequences

The loader returns numpy arrays and a TelemetryWindowDataset ready for PyTorch.
"""

import os
import json
import ast
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from torch.utils.data import DataLoader

from data.preprocessing import (
    ChannelScaler,
    TelemetryWindowDataset,
    time_split,
    build_sensor_graph,
)


# ──────────────────────────────────────────────────────────────
# Label utilities
# ──────────────────────────────────────────────────────────────

def _parse_anomaly_sequences(sequences_str: str, length: int) -> np.ndarray:
    """Convert anomaly_sequences string like '[[0, 100], [200, 250]]' to label array."""
    labels = np.zeros(length, dtype=np.int64)
    try:
        seqs = ast.literal_eval(sequences_str)
        for start, end in seqs:
            labels[start : end + 1] = 1
    except Exception:
        pass
    return labels


# ──────────────────────────────────────────────────────────────
# Main loader
# ──────────────────────────────────────────────────────────────

class SMAPMSLLoader:
    """
    Loads one channel (entity) from the SMAP or MSL dataset.

    Parameters
    ----------
    root_dir  : path to data/raw/SMAP_MSL/
    channel   : channel id, e.g. "P-1" (SMAP) or "C-1" (MSL)
    spacecraft: "SMAP" or "MSL"
    window_size, stride : sliding window parameters
    normalize : whether to z-score normalise
    """

    LABEL_FILE = "labeled_anomalies.csv"

    def __init__(
        self,
        root_dir: str = "data/raw/SMAP_MSL",
        channel: str = "P-1",
        spacecraft: str = "SMAP",
        window_size: int = 128,
        stride: int = 1,
        normalize: bool = True,
    ):
        self.root_dir = root_dir
        self.channel = channel
        self.spacecraft = spacecraft
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.scaler = ChannelScaler()

        self._load()

    # ----------------------------------------------------------
    def _load(self):
        train_path = os.path.join(self.root_dir, "train", f"{self.channel}.npy")
        test_path  = os.path.join(self.root_dir, "test",  f"{self.channel}.npy")
        label_path = os.path.join(self.root_dir, self.LABEL_FILE)

        # --- raw arrays ---
        if os.path.exists(train_path):
            self.train_raw = np.load(train_path).astype(np.float32)   # (T_tr, C)
            self.test_raw  = np.load(test_path).astype(np.float32)    # (T_te, C)
        else:
            print(f"[SMAP/MSL] Data files not found at {train_path}.")
            print("  → Generating synthetic placeholder data for development.")
            self.train_raw, self.test_raw = self._synthetic_placeholder()

        # ensure 2D
        if self.train_raw.ndim == 1:
            self.train_raw = self.train_raw[:, None]
            self.test_raw  = self.test_raw[:, None]

        self.n_channels = self.train_raw.shape[1]

        # --- labels ---
        if os.path.exists(label_path):
            df = pd.read_csv(label_path)
            row = df[(df["chan_id"] == self.channel) &
                     (df["spacecraft"] == self.spacecraft)]
            if len(row) > 0:
                seqs = row.iloc[0]["anomaly_sequences"]
                self.test_labels = _parse_anomaly_sequences(seqs, len(self.test_raw))
            else:
                self.test_labels = np.zeros(len(self.test_raw), dtype=np.int64)
        else:
            print("[SMAP/MSL] labeled_anomalies.csv not found → using zero labels.")
            self.test_labels = np.zeros(len(self.test_raw), dtype=np.int64)

        self.train_labels = np.zeros(len(self.train_raw), dtype=np.int64)

        # --- normalise ---
        if self.normalize:
            self.train_norm = self.scaler.fit_transform(self.train_raw)
            self.test_norm  = self.scaler.transform(self.test_raw)
        else:
            self.train_norm = self.train_raw
            self.test_norm  = self.test_raw

    # ----------------------------------------------------------
    def _synthetic_placeholder(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic sine-wave telemetry for offline development."""
        rng = np.random.default_rng(0)
        T_tr, T_te, C = 5000, 2000, 25
        t_tr = np.linspace(0, 20 * np.pi, T_tr)
        t_te = np.linspace(0, 8 * np.pi, T_te)
        freq = rng.uniform(0.5, 3.0, C)
        train = (np.sin(t_tr[:, None] * freq[None, :]) +
                 rng.normal(0, 0.05, (T_tr, C))).astype(np.float32)
        test  = (np.sin(t_te[:, None] * freq[None, :]) +
                 rng.normal(0, 0.05, (T_te, C))).astype(np.float32)
        # inject a few spike anomalies in the test set
        for c in rng.integers(0, C, 5):
            idx = rng.integers(100, T_te - 100)
            test[idx:idx+20, c] += 5.0
        return train, test

    # ----------------------------------------------------------
    def get_datasets(self) -> Tuple[TelemetryWindowDataset, TelemetryWindowDataset]:
        train_ds = TelemetryWindowDataset(
            self.train_norm, self.train_labels, self.window_size, self.stride
        )
        test_ds = TelemetryWindowDataset(
            self.test_norm, self.test_labels, self.window_size, self.stride
        )
        return train_ds, test_ds

    def get_loaders(
        self, batch_size: int = 64, num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader]:
        train_ds, test_ds = self.get_datasets()
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True,  num_workers=num_workers)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers)
        return train_loader, test_loader

    def get_sensor_graph(self, threshold: float = 0.3):
        return build_sensor_graph(self.train_norm, threshold)

    # ----------------------------------------------------------
    @property
    def channel_names(self) -> List[str]:
        return [f"{self.channel}_ch{i}" for i in range(self.n_channels)]

    def summary(self):
        n_anom = self.test_labels.sum()
        pct    = 100 * n_anom / len(self.test_labels)
        print(f"[SMAP/MSL] channel={self.channel}  spacecraft={self.spacecraft}")
        print(f"  Train: {self.train_raw.shape}  |  Test: {self.test_raw.shape}")
        print(f"  Channels: {self.n_channels}  |  "
              f"Test anomalies: {n_anom} ({pct:.1f}%)")


# ──────────────────────────────────────────────────────────────
# Helper: list all channels in a dataset directory
# ──────────────────────────────────────────────────────────────

def list_channels(root_dir: str = "data/raw/SMAP_MSL",
                  spacecraft: str = "SMAP") -> List[str]:
    """Return channel IDs present in the train/ folder."""
    train_dir = os.path.join(root_dir, "train")
    if not os.path.exists(train_dir):
        return []
    label_path = os.path.join(root_dir, "labeled_anomalies.csv")
    if os.path.exists(label_path):
        df = pd.read_csv(label_path)
        return df[df["spacecraft"] == spacecraft]["chan_id"].tolist()
    return [f.replace(".npy", "") for f in os.listdir(train_dir)
            if f.endswith(".npy")]
