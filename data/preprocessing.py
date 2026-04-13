"""
data/preprocessing.py — Shared preprocessing utilities for telemetry time series.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────
# Normalisation
# ──────────────────────────────────────────────────────────────

class ChannelScaler:
    """Z-score normaliser that fits per-channel and is pickle-friendly."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: np.ndarray) -> "ChannelScaler":
        # X: (T, C)
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(X)


# ──────────────────────────────────────────────────────────────
# Sliding-Window Dataset
# ──────────────────────────────────────────────────────────────

class TelemetryWindowDataset(Dataset):
    """
    Converts a (T, C) telemetry array + (T,) label array into
    overlapping windows of shape (window_size, C).
    Label for a window = 1 if ANY step inside it is anomalous.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        window_size: int = 128,
        stride: int = 1,
    ):
        super().__init__()
        self.window_size = window_size
        self.stride = stride

        windows, window_labels = [], []
        T = len(data)
        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            windows.append(data[start:end])
            window_labels.append(int(labels[start:end].any()))

        self.X = torch.tensor(np.array(windows), dtype=torch.float32)  # (N, W, C)
        self.y = torch.tensor(window_labels, dtype=torch.long)          # (N,)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ──────────────────────────────────────────────────────────────
# Missing-value handling
# ──────────────────────────────────────────────────────────────

def fill_missing(df: pd.DataFrame, method: str = "linear") -> pd.DataFrame:
    """Interpolate or forward-fill missing values."""
    if method == "linear":
        return df.interpolate(method="linear", limit_direction="both")
    elif method == "ffill":
        return df.fillna(method="ffill").fillna(method="bfill")
    else:
        return df.fillna(0.0)


# ──────────────────────────────────────────────────────────────
# Train / test split (time-aware — no leakage)
# ──────────────────────────────────────────────────────────────

def time_split(
    data: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split time series chronologically."""
    cut = int(len(data) * train_ratio)
    return data[:cut], labels[:cut], data[cut:], labels[cut:]


# ──────────────────────────────────────────────────────────────
# Correlation-based graph construction (for GNN)
# ──────────────────────────────────────────────────────────────

def build_sensor_graph(
    data: np.ndarray,
    threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (edge_index, edge_weight) for a sensor correlation graph.
    data: (T, C) — each column is a sensor.
    Edges connect pairs whose |correlation| > threshold.
    """
    C = data.shape[1]
    corr = np.corrcoef(data.T)  # (C, C)
    src, dst, weights = [], [], []
    for i in range(C):
        for j in range(C):
            if i != j and abs(corr[i, j]) > threshold:
                src.append(i)
                dst.append(j)
                weights.append(float(abs(corr[i, j])))

    edge_index = np.array([src, dst], dtype=np.int64)   # (2, E)
    edge_weight = np.array(weights, dtype=np.float32)   # (E,)
    return edge_index, edge_weight


# ──────────────────────────────────────────────────────────────
# Synthetic anomaly injection (for self-supervised training)
# ──────────────────────────────────────────────────────────────

def inject_anomalies(
    data: np.ndarray,
    anomaly_ratio: float = 0.05,
    anomaly_types: List[str] = ("spike", "shift", "noise"),
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject synthetic anomalies into clean telemetry for self-supervised learning.
    Returns augmented data and binary label array.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    data = data.copy()
    T, C = data.shape
    labels = np.zeros(T, dtype=np.int64)
    n_anomaly_points = max(1, int(T * anomaly_ratio))

    for _ in range(n_anomaly_points):
        t = rng.integers(0, T)
        c = rng.integers(0, C)
        atype = rng.choice(list(anomaly_types))

        if atype == "spike":
            data[t, c] += rng.uniform(3, 7) * data[:, c].std() * rng.choice([-1, 1])
        elif atype == "shift":
            length = rng.integers(5, 30)
            end = min(T, t + length)
            data[t:end, c] += rng.uniform(2, 5) * data[:, c].std()
            labels[t:end] = 1
        elif atype == "noise":
            length = rng.integers(5, 20)
            end = min(T, t + length)
            data[t:end, c] += rng.normal(0, data[:, c].std() * 2, size=end - t)
            labels[t:end] = 1

        labels[t] = 1

    return data, labels


# ──────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────

def point_adjust(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """
    Point-adjust strategy: if any point in an anomaly segment is detected,
    all points in that segment are counted as detected (standard in AD literature).
    """
    adjusted = preds.copy()
    in_anomaly = False
    seg_detected = False
    seg_start = 0

    for t in range(len(labels)):
        if labels[t] == 1 and not in_anomaly:
            in_anomaly = True
            seg_start = t
            seg_detected = False
        if in_anomaly and preds[t] == 1:
            seg_detected = True
        if (labels[t] == 0 or t == len(labels) - 1) and in_anomaly:
            if seg_detected:
                adjusted[seg_start:t] = 1
            in_anomaly = False

    return adjusted


def detection_delay(labels: np.ndarray, preds: np.ndarray) -> float:
    """
    Average number of steps between anomaly onset and first detection.
    Returns np.inf if no anomaly is ever detected.
    """
    delays = []
    in_anomaly = False
    onset = 0

    for t in range(len(labels)):
        if labels[t] == 1 and not in_anomaly:
            in_anomaly = True
            onset = t
        if in_anomaly and preds[t] == 1:
            delays.append(t - onset)
            in_anomaly = False
        if labels[t] == 0 and in_anomaly:
            in_anomaly = False

    return float(np.mean(delays)) if delays else float("inf")
