# Spacecraft Anomaly Detection — Explainable Multi-Modal Digital Twin

A full end-to-end research pipeline for detecting anomalies in spacecraft telemetry,
combining deep learning, graph neural networks, multi-modal fusion, explainability,
and a generative digital twin visualiser.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT STREAMS                               │
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
                    │    WEB APP      │
                    │ (FastAPI UI/API)│
                    └─────────────────┘
```

---

## Project Structure

```
XAI-SPACECRAFT-ANOMALY-DETECTION-MULTI-MODEL-SYSTEM/
├── src/spacecraft_anomaly/     # Installable Python package
│   ├── config.py               # Central configuration dataclasses
│   ├── paths.py                # Project-root path resolution
│   ├── training.py             # Model factory + train loop
│   ├── data/                   # Loaders and preprocessing
│   ├── models/                 # LSTM, Transformer, GNN, Fusion, Digital Twin
│   ├── detection/              # Threshold calibration and metrics
│   └── explainability/         # SHAP, attention, causal graph
├── scripts/                    # CLI entry points
│   ├── train.py
│   ├── evaluate.py
│   ├── run_pipeline.py         # End-to-end demo
│   └── download_smap_data.ps1  # SMAP/MSL dataset (HF or Kaggle)
├── data/raw/                   # Datasets only (not importable as a package)
│   ├── SMAP_MSL/
│   └── OPSSAT/
├── checkpoints/                # Saved model weights (*.pt)
├── results/                    # Evaluation outputs (gitignored artifacts)
├── webapp/                     # FastAPI + Jinja2 web UI/API (no Streamlit)
├── requirements.txt
└── pyproject.toml
```

---

## Quick Start

### 1. Install dependencies

```powershell
cd D:\ProgramXXX\XAI-SPACECRAFT-ANOMALY-DETECTION-MULTI-MODEL-SYSTEM
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Optional GNN support is included via `torch-geometric` in `requirements.txt`.

### 2. Run the demo (no data download needed)

```powershell
python scripts\run_pipeline.py --model GNN --epochs 5
```

Outputs are written to `demo_output/` (gitignored).

### 3. Download real datasets

#### NASA SMAP/MSL

The loader expects this layout (paths relative to the repo root):

```
data/raw/SMAP_MSL/
├── labeled_anomalies.csv      # chan_id, spacecraft, anomaly_sequences, ...
├── train/
│   ├── P-1.npy                # (T_train, C) float32 per channel
│   └── ...
└── test/
    ├── P-1.npy                # (T_test, C)
    └── ...
```

**Troubleshooting:** The old Telemanom S3 URL (`https://s3-us-west-2.amazonaws.com/telemanom/data.zip`) no longer allows public downloads (403 AccessDenied). Tutorials that still reference it will fail at `Expand-Archive` because `data.zip` was never created. Use the mirrors below instead.

**Quick install (recommended, PowerShell from repo root):**

```powershell
.\scripts\download_smap_data.ps1
```

Downloads all 82 channels from [Hugging Face `appleparan/telemanom`](https://huggingface.co/datasets/appleparan/telemanom) plus `labeled_anomalies.csv` from GitHub. Re-run is safe (existing files are skipped).

**Option A — Hugging Face (no Kaggle account, ~250 MB total)**

Same as the quick install: `.\scripts\download_smap_data.ps1` (default `-Source huggingface`).

Bulk alternative if you have the Hugging Face CLI (`pip install huggingface_hub`):

```powershell
hf download appleparan/telemanom --repo-type dataset --local-dir data\raw\SMAP_MSL\_hf
# Then move _hf\data\data\train -> data\raw\SMAP_MSL\train, _hf\data\data\test -> test, and copy labeled_anomalies.csv to data\raw\SMAP_MSL\
```

**Option B — Kaggle (single zip, official mirror in [khundman/telemanom](https://github.com/khundman/telemanom))**

Requires a free [Kaggle](https://www.kaggle.com/) account and API token in `%USERPROFILE%\.kaggle\kaggle.json` ([setup guide](https://www.kaggle.com/docs/api#authentication)).

```powershell
pip install kaggle
.\scripts\download_smap_data.ps1 -Source kaggle
```

Manual equivalent (bash-style steps adapted for PowerShell):

```powershell
$dest = "data\raw\SMAP_MSL"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Set-Location $dest
kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl
Expand-Archive -Force nasa-anomaly-detection-dataset-smap-msl.zip -DestinationPath .
Move-Item -Force data\data\train train
Move-Item -Force data\data\test test
Remove-Item -Recurse -Force data
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv" -OutFile labeled_anomalies.csv
Set-Location ..\..\..
```

Dataset page: [NASA Anomaly Detection Dataset SMAP & MSL](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl)

**Option C — bash (Linux/macOS/WSL)**

```bash
pip install kaggle
mkdir -p data/raw/SMAP_MSL && cd data/raw/SMAP_MSL
kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl
unzip -o nasa-anomaly-detection-dataset-smap-msl.zip
mv data/data/train train && mv data/data/test test && rm -rf data *.zip
curl -L -o labeled_anomalies.csv https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
cd ../../..
```

**Verify:**

```powershell
python -c "from spacecraft_anomaly.data.smap_msl import list_channels; print('channels:', len(list_channels()))"
python scripts\train.py --model GNN --dataset SMAP --channel P-1 --epochs 1
```

You should see ~55 SMAP + ~27 MSL channel IDs and no “synthetic placeholder” message from the loader.

#### ESA OPS-SAT-AD

Download from [Zenodo 7937210](https://zenodo.org/record/7937210) and extract into `data\raw\OPSSAT\`.

### 4. Train a model

```powershell
python scripts\train.py --model GNN --dataset SMAP --channel P-1 --epochs 50
python scripts\train.py --model TRANSFORMER --dataset OPSSAT --channel 1 --epochs 30
python scripts\train.py --model LSTM_AE --dataset MSL --channel C-1 --epochs 50
```

Checkpoints are saved under `checkpoints\`.

### 5. Evaluate

```powershell
python scripts\evaluate.py --checkpoint checkpoints\GNN_SMAP_P-1_best.pt --dataset SMAP --channel P-1
```

Plots and score arrays are saved under `results\`.

### 6. Launch the web app

```powershell
uvicorn webapp.main:app --reload
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

```python
from spacecraft_anomaly.models.digital_twin import generate_synthetic_spacecraft_image

img = generate_synthetic_spacecraft_image(anomalous=True, subsystem=0, severity=0.8)
```

Or run directly from the command line:

```powershell
python -c "from spacecraft_anomaly.models.digital_twin import generate_synthetic_spacecraft_image; img = generate_synthetic_spacecraft_image(anomalous=True, subsystem=0, severity=0.8); print('Generated image shape:', img.shape)"
```

---

## Configuration

All hyperparameters live in `src/spacecraft_anomaly/config.py`. Enter the Python REPL and run:

```python
from spacecraft_anomaly.config import Config
cfg = Config()
cfg.model.model_type = "GNN"
cfg.train.epochs = 50
cfg.detection.threshold_percentile = 99.5
```

Or from the command line:

```powershell
python -c "from spacecraft_anomaly.config import Config; cfg = Config(); cfg.model.model_type = 'GNN'; cfg.train.epochs = 50; cfg.detection.threshold_percentile = 99.5; print(cfg)"
```

Dataset paths are relative to the repository root and resolved automatically.

---

## Citation / References

- STGLR (2025): Dynamic inter-sensor graph with GraphSAGE+GRU+VAE
- TranAD (Tuli et al., 2022): Two-stage focus transformer
- OPS-SAT-AD: ESA anomaly detection benchmark
- Iino et al. (2024): FRAM-based explainability on ISS telemetry
