"""
config.py — Central configuration for the Spacecraft Anomaly Detection pipeline.
Edit these settings to switch datasets, model types, or training parameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    # --- Dataset selection ---
    dataset: str = "SMAP"          # "SMAP" | "MSL" | "OPSSAT"
    channel: str = "P-1"           # telemetry channel / entity name
    window_size: int = 128         # sliding window length (time steps)
    stride: int = 1                # sliding window stride
    train_ratio: float = 0.8       # fraction of data used for training
    normalize: bool = True         # z-score normalisation per channel

    # --- Paths (relative to project root) ---
    smap_msl_dir: str = "data/raw/SMAP_MSL"
    opssat_dir: str = "data/raw/OPSSAT"
    processed_dir: str = "data/processed"


@dataclass
class ModelConfig:
    model_type: str = "GNN"        # "LSTM_AE" | "TRANSFORMER" | "GNN" | "FUSION"
    input_dim: int = 25            # number of telemetry channels
    hidden_dim: int = 32
    latent_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4             # for Transformer
    dropout: float = 0.1
    use_image_branch: bool = False # enable CNN image encoder in fusion mode

    # GNN-specific
    gnn_type: str = "SAGE"        # "SAGE" | "GAT" | "GCN"
    edge_threshold: float = 0.3   # correlation threshold for graph construction

    # Digital Twin (VAE generator)
    twin_image_size: int = 64      # generated image resolution


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10             # early stopping patience
    device: str = "auto"           # "auto" | "cpu" | "cuda" | "mps"
    seed: int = 42
    log_interval: int = 10        # log every N batches
    checkpoint_dir: str = "checkpoints"
    use_amp: bool = False          # automatic mixed precision (GPU only)


@dataclass
class DetectionConfig:
    # Threshold strategy
    threshold_method: str = "percentile"  # "percentile" | "std" | "fixed"
    threshold_percentile: float = 99.0
    threshold_std_factor: float = 3.0
    fixed_threshold: float = 0.5

    # Dual-stage pipeline
    use_dual_stage: bool = True
    fast_model: str = "mahalanobis"   # "mahalanobis" | "isolation_forest"
    deep_model: str = "GNN"

    # Alert smoothing
    alert_window: int = 10             # majority vote over N steps


@dataclass
class ExplainConfig:
    use_shap: bool = True
    shap_background_samples: int = 100
    use_attention: bool = True
    use_causal: bool = True
    causal_max_lag: int = 5            # Granger causality max lag
    top_k_features: int = 10          # show top-k contributing features


@dataclass
class AppConfig:
    host: str = "localhost"
    port: int = 8501
    refresh_interval: float = 1.0     # seconds between telemetry updates (demo)
    theme: str = "dark"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    app: AppConfig = field(default_factory=AppConfig)


# --- Convenience singleton ---
DEFAULT_CONFIG = Config()
