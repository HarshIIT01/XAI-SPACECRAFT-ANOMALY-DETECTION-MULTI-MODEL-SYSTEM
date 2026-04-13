# Spacecraft Anomaly Detection — Explainable Multi-Modal Digital Twin

A full end-to-end research pipeline for detecting anomalies in spacecraft telemetry,
combining deep learning, graph neural networks, multi-modal fusion, explainability,
and a generative digital twin visualiser.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT STREAMS                               |
│  Telemetry (T×C)          Spacecraft Image (3×H×W)                  │
└────────────┬──────────────────────────┬─────────────────────────────┘
             │                          │
     ┌───────▼────────┐        ┌────────▼────────┐
     │  GNN + GRU     │        │  CNN Encoder    │
     │  (inter-sensor │        │  (ResNet-lite)  │
     │   graph VAE)   │        │                 │
     └───────┬────────┘        └────────┬────────┘
             │   z_tel                  │  z_img
             └──────────┬───────────────┘
                        │
               ┌────────▼──────────┐
               │  Cross-Attention  │
               │  Fusion Layer     │
               └────────┬──────────┘
                        │  z_fused
           ┌────────────┴──────────────────┐
           │                               │
   ┌───────▼────────┐            ┌─────────▼──────────┐
   │  Anomaly Head  │            │  Digital Twin VAE   │
   │  (VAE + BCE)   │            │  (Cond. image gen.) │
   └───────┬────────┘            └─────────┬──────────┘
           │                               │
   ┌───────▼──────────────────────────────▼──────────┐
   │             EXPLAINABILITY LAYER                 │
   │   SHAP attribution  ·  Attention heatmap         │
   │   Causal graph      ·  NL report                 │
   └─────────────────────────┬────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   DASHBOARD     │
                    │   (Streamlit)   │
                    └─────────────────┘
```

---

## Project Structure

```
spacecraft_anomaly/
├── config.py                    # Central configuration dataclasses
├── requirements.txt             # All Python dependencies
├── train.py                     # Training CLI
├── evaluate.py                  # Evaluation + plots CLI
├── run_pipeline.py              # Full end-to-end demo
│
├── data/
│   ├── preprocessing.py         # Normalisation, windowing, graph builder
│   ├── smap_msl.py             # NASA SMAP/MSL data loader
│   └── opssat.py               # ESA OPS-SAT-AD data loader
│
├── models/
│   ├── lstm_ae.py              # LSTM Autoencoder + VAE variant
│   ├── transformer_ad.py       # TranAD-style Transformer + PatchTransformer
│   ├── graph_rnn.py            # GraphSAGE + GRU + VAE (STGLR-inspired)
│   ├── fusion.py               # Multi-modal CNN + telemetry fusion
│   └── digital_twin.py         # Conditional VAE image generator
│
├── explainability/
│   └── __init__.py             # SHAP, Attention, Causal Graph
│
├── detection/
│   └── detector.py             # Threshold calibration, dual-stage pipeline
│
└── dashboard/
    └── app.py                  # Streamlit real-time monitoring dashboard
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# For GNN support:
pip install torch-geometric
```

### 2. Run the demo (no data download needed)

```bash
python run_pipeline.py --model GNN --dataset OPSSAT --channel 1 --epochs 5
# Output in demo_output/
```

For SMAP/MSL, use:

```bash
python run_pipeline.py --model GNN --dataset SMAP --channel P-1 --epochs 5
```

### 3. Download real datasets

**NASA SMAP/MSL:**
```bash
mkdir -p data/raw/SMAP_MSL
cd data/raw/SMAP_MSL
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip
unzip data.zip
wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
```

**ESA OPS-SAT-AD:**
```bash
# Download from: https://zenodo.org/record/7937210
mkdir -p data/raw/OPSSAT
# Extract into data/raw/OPSSAT/
```

### 4. Train a model

```bash
# GNN model on SMAP, channel P-1
python train.py --model GNN --dataset SMAP --channel P-1 --epochs 50

# Transformer on OPS-SAT
python train.py --model TRANSFORMER --dataset OPSSAT --channel 1 --epochs 30

# LSTM Autoencoder on MSL
python train.py --model LSTM_AE --dataset MSL --channel C-1 --epochs 50
```

### 5. Evaluate

```bash
python evaluate.py \
  --checkpoint checkpoints/GNN_SMAP_P-1_best.pt \
  --dataset SMAP --channel P-1
```

### 6. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

## Models

| Model | Type | Key Strength |
|-------|------|-------------|
| `LSTM_AE` | Reconstruction AE | Fast, handles temporal patterns |
| `LSTM_VAE` | Variational AE | Structured latent, better uncertainty |
| `TRANSFORMER` | TranAD-style | Long-range dependencies, attention maps |
| `GNN` | GraphSAGE+GRU+VAE | Inter-sensor correlations, best F1 |
| `FUSION` | Multimodal | Telemetry + imagery jointly |

---

## Explainability Layers

1. **SHAP Feature Attribution** — Which sensor channels contributed most to the anomaly score.
2. **Temporal Attention** — Which time steps within the window were most anomalous.
3. **Causal Graph** — Granger-causality DAG tracing anomaly to root sensor upstream.
4. **Natural Language Report** — Human-readable alert with recommended actions.

---

## Digital Twin

The digital twin generates a synthetic spacecraft image conditioned on the telemetry embedding and anomaly severity. The affected subsystem (solar panels, main body, antenna) is highlighted in red:

```python
from models.digital_twin import generate_synthetic_spacecraft_image
img = generate_synthetic_spacecraft_image(anomalous=True, subsystem=0, severity=0.8)
```

---

## Datasets

| Dataset | Samples | Channels | Anomaly % | Reference |
|---------|---------|----------|-----------|-----------|
| NASA SMAP | ~135k | 25 | ~13% | Hundman et al. 2018 |
| NASA MSL | ~58k | 55 | ~10% | Hundman et al. 2018 |
| ESA OPS-SAT-AD | ~2.1k frags | 9 | ~20% | ESA 2023 |

---

## Configuration

All hyperparameters are centralised in `config.py`:

```python
from config import Config
cfg = Config()
cfg.model.model_type = "GNN"
cfg.train.epochs = 50
cfg.detection.threshold_percentile = 99.5
```

---

## Citation / References

- STGLR (2025): Dynamic inter-sensor graph with GraphSAGE+GRU+VAE
- TranAD (Tuli et al., 2022): Two-stage focus transformer
- OPS-SAT-AD: ESA anomaly detection benchmark
- Iino et al. (2024): FRAM-based explainability on ISS telemetry
