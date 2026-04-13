"""
data/opssat.py — Loader for ESA's OPS-SAT Anomaly Detection benchmark.

Dataset homepage: https://zenodo.org/record/7937210
Expected layout after extraction:

  data/raw/OPSSAT/
      train/   <channel_id>/   *.csv  (normal segments)
      test/    <channel_id>/   *.csv  (mix normal + anomalous)
      labels/  <channel_id>/   *.csv  (columns: start, end, anomaly)
      images/  <channel_id>/   *.png  (optional Earth-obs thumbnails)

If files are absent, a synthetic placeholder is generated so that the
rest of the pipeline can be developed and tested locally.
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from torch.utils.data import DataLoader, Dataset
import torch

from data.preprocessing import (
    ChannelScaler,
    TelemetryWindowDataset,
    fill_missing,
    build_sensor_graph,
)


# ──────────────────────────────────────────────────────────────
# Image dataset (optional multimodal branch)
# ──────────────────────────────────────────────────────────────

class OpsSatImageDataset(Dataset):
    """
    Pairs each telemetry window with the closest-in-time satellite image
    (if available).  Falls back to a blank tensor when images are absent.
    """

    def __init__(
        self,
        telemetry_dataset: TelemetryWindowDataset,
        image_dir: Optional[str] = None,
        image_size: int = 64,
    ):
        from torchvision import transforms
        self.tel_ds = telemetry_dataset
        self.image_dir = image_dir
        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        self.image_paths: List[str] = []
        if image_dir and os.path.isdir(image_dir):
            self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))

    def __len__(self) -> int:
        return len(self.tel_ds)

    def __getitem__(self, idx):
        x, y = self.tel_ds[idx]
        if self.image_paths:
            # nearest image by index (very simplified alignment)
            img_idx = min(idx, len(self.image_paths) - 1)
            from PIL import Image
            img = Image.open(self.image_paths[img_idx]).convert("RGB")
            img_tensor = self.img_transform(img)   # (3, H, W)
        else:
            img_tensor = torch.zeros(3, 64, 64)
        return x, img_tensor, y


# ──────────────────────────────────────────────────────────────
# Main loader
# ──────────────────────────────────────────────────────────────

class OpsSatLoader:
    """
    Loads one OPS-SAT telemetry channel for anomaly detection.

    Parameters
    ----------
    root_dir    : path to data/raw/OPSSAT/
    channel_id  : e.g. "1" … "9"
    window_size : sliding window length
    stride      : window stride
    normalize   : z-score per channel
    include_images : whether to build image-paired dataset
    """

    CHANNELS = [str(i) for i in range(1, 10)]   # 9 channels

    def __init__(
        self,
        root_dir: str = "data/raw/OPSSAT",
        channel_id: str = "1",
        window_size: int = 128,
        stride: int = 1,
        normalize: bool = True,
        include_images: bool = False,
    ):
        self.root_dir = root_dir
        self.channel_id = channel_id
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.include_images = include_images
        self.scaler = ChannelScaler()
        self._load()

    # ----------------------------------------------------------
    def _load(self):
        train_dir  = os.path.join(self.root_dir, "train", self.channel_id)
        test_dir   = os.path.join(self.root_dir, "test",  self.channel_id)
        label_dir  = os.path.join(self.root_dir, "labels", self.channel_id)
        self.image_dir = os.path.join(self.root_dir, "images", self.channel_id)

        if os.path.isdir(train_dir):
            self.train_raw, self.train_labels = self._load_split(train_dir, label_dir, split="train")
            self.test_raw,  self.test_labels  = self._load_split(test_dir,  label_dir, split="test")
        else:
            print(f"[OPS-SAT] Data not found at {train_dir}.")
            print("  → Generating synthetic placeholder data for development.")
            self.train_raw, self.train_labels, \
            self.test_raw,  self.test_labels   = self._synthetic_placeholder()

        self.n_channels = self.train_raw.shape[1]

        if self.normalize:
            self.train_norm = self.scaler.fit_transform(self.train_raw)
            self.test_norm  = self.scaler.transform(self.test_raw)
        else:
            self.train_norm = self.train_raw.copy()
            self.test_norm  = self.test_raw.copy()

    # ----------------------------------------------------------
    def _load_split(
        self, data_dir: str, label_dir: str, split: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        frames = []
        for f in csv_files:
            df = pd.read_csv(f)
            frames.append(df)
        if not frames:
            raise FileNotFoundError(f"No CSV files in {data_dir}")

        data_df = pd.concat(frames, ignore_index=True)
        data_df = fill_missing(data_df)

        # Drop non-numeric & timestamp columns
        num_cols = data_df.select_dtypes(include=np.number).columns.tolist()
        data = data_df[num_cols].values.astype(np.float32)

        # Labels
        T = len(data)
        labels = np.zeros(T, dtype=np.int64)
        if os.path.isdir(label_dir):
            lbl_files = sorted(glob.glob(os.path.join(label_dir, "*.csv")))
            offset = 0
            for f in lbl_files:
                ldf = pd.read_csv(f)
                if set(["start", "end", "anomaly"]).issubset(ldf.columns):
                    for _, row in ldf.iterrows():
                        if row["anomaly"] == 1:
                            s = int(row["start"]) + offset
                            e = int(row["end"])   + offset
                            labels[s:e+1] = 1
                offset += len(pd.read_csv(f))  # accumulate offset

        return data, labels

    # ----------------------------------------------------------
    def _synthetic_placeholder(self):
        rng = np.random.default_rng(1)
        T_tr, T_te, C = 4000, 1500, 9
        t_tr = np.linspace(0, 16 * np.pi, T_tr)
        t_te = np.linspace(0, 6  * np.pi, T_te)
        freq = rng.uniform(0.5, 2.5, C)
        train = (np.sin(t_tr[:, None] * freq) +
                 rng.normal(0, 0.1, (T_tr, C))).astype(np.float32)
        test  = (np.sin(t_te[:, None] * freq) +
                 rng.normal(0, 0.1, (T_te, C))).astype(np.float32)

        tr_labels = np.zeros(T_tr, dtype=np.int64)
        te_labels = np.zeros(T_te, dtype=np.int64)
        # inject anomalies in test
        for _ in range(8):
            c   = rng.integers(0, C)
            idx = rng.integers(50, T_te - 80)
            length = rng.integers(10, 60)
            test[idx:idx+length, c] += rng.uniform(3, 6) * rng.choice([-1, 1])
            te_labels[idx:idx+length] = 1

        return train, tr_labels, test, te_labels

    # ----------------------------------------------------------
    def get_datasets(
        self,
    ) -> Tuple[TelemetryWindowDataset, TelemetryWindowDataset]:
        train_ds = TelemetryWindowDataset(
            self.train_norm, self.train_labels, self.window_size, self.stride
        )
        test_ds = TelemetryWindowDataset(
            self.test_norm, self.test_labels, self.window_size, self.stride
        )
        return train_ds, test_ds

    def get_multimodal_datasets(
        self, image_size: int = 64
    ) -> Tuple[OpsSatImageDataset, OpsSatImageDataset]:
        tr_ds, te_ds = self.get_datasets()
        img_dir = self.image_dir if self.include_images else None
        return (OpsSatImageDataset(tr_ds, img_dir, image_size),
                OpsSatImageDataset(te_ds, img_dir, image_size))

    def get_loaders(
        self, batch_size: int = 64, num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader]:
        train_ds, test_ds = self.get_datasets()
        return (DataLoader(train_ds, batch_size=batch_size,
                           shuffle=True, num_workers=num_workers),
                DataLoader(test_ds,  batch_size=batch_size,
                           shuffle=False, num_workers=num_workers))

    def get_sensor_graph(self, threshold: float = 0.3):
        return build_sensor_graph(self.train_norm, threshold)

    def summary(self):
        n_anom = self.test_labels.sum()
        pct    = 100 * n_anom / len(self.test_labels)
        print(f"[OPS-SAT] channel={self.channel_id}")
        print(f"  Train: {self.train_raw.shape}  |  Test: {self.test_raw.shape}")
        print(f"  Channels: {self.n_channels}  |  "
              f"Test anomalies: {n_anom} ({pct:.1f}%)")
