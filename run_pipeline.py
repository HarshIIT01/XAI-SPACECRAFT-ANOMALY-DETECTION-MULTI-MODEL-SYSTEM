"""
run_pipeline.py — End-to-end demonstration of the full pipeline.

No real dataset required: uses synthetic placeholder data.
Runs all components: data loading → training → evaluation → explainability → digital twin.

Usage
-----
  python run_pipeline.py
  python run_pipeline.py --model TRANSFORMER --epochs 10
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from data.smap_msl import SMAPMSLLoader
from data.preprocessing import inject_anomalies
from train import build_model, get_device, train_epoch
from detection.detector import ThresholdCalibrator, evaluate, print_metrics
from explainability import SHAPExplainer, AttentionVisualiser, CausalGraph
from models.digital_twin import DigitalTwin, generate_synthetic_spacecraft_image
from torch.optim import AdamW


# ──────────────────────────────────────────────────────────────

def run_demo(model_type: str = "GNN", epochs: int = 5, out_dir: str = "demo_output"):
    os.makedirs(out_dir, exist_ok=True)
    cfg    = Config()
    device = get_device("auto")

    print("=" * 65)
    print("  SPACECRAFT ANOMALY DETECTION — FULL PIPELINE DEMO")
    print("=" * 65)

    # ── 1. Data ───────────────────────────────────────────────
    print("\n[1/6] Loading data (synthetic placeholder)...")
    data_loader = SMAPMSLLoader(window_size=128)
    data_loader.summary()
    train_dl, test_dl = data_loader.get_loaders(batch_size=32)
    n_channels  = data_loader.n_channels
    channel_names = data_loader.channel_names

    # ── 2. Model ──────────────────────────────────────────────
    print(f"\n[2/6] Building {model_type} model...")
    model = build_model(model_type, n_channels, cfg.model, seq_len=128).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=1e-3)

    # ── 3. Training ───────────────────────────────────────────
    print(f"\n[3/6] Training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_dl, optimizer, device, model_type, epoch, epochs)
        print(f"  Epoch {epoch}/{epochs}  |  loss={loss:.5f}")

    # ── 4. Evaluation ─────────────────────────────────────────
    print("\n[4/6] Evaluating on test set...")
    model.eval()

    # Get training scores for threshold calibration
    train_scores = []
    with torch.no_grad():
        for x, _ in train_dl:
            s = model.anomaly_score(x.to(device)).cpu().numpy()
            train_scores.append(s)
    train_scores = np.concatenate(train_scores)

    calibrator = ThresholdCalibrator("percentile", 99.0)
    calibrator.fit(train_scores)

    test_scores, test_labels = [], []
    test_windows = []
    with torch.no_grad():
        for x, y in test_dl:
            s = model.anomaly_score(x.to(device)).cpu().numpy()
            test_scores.append(s)
            test_labels.append(y.numpy())
            test_windows.append(x.cpu().numpy())

    test_scores  = np.concatenate(test_scores)
    test_labels  = np.concatenate(test_labels)
    test_windows = np.concatenate(test_windows)

    metrics = evaluate(test_scores, test_labels, calibrator.threshold)
    print_metrics(metrics, f"{model_type} — Evaluation")

    # Score timeline plot
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(test_scores, linewidth=0.8, color="#3498db", label="Score")
    ax.axhline(calibrator.threshold, color="#e74c3c", linestyle="--", label="Threshold")
    if test_labels.sum() > 0:
        anom_idx = np.where(test_labels == 1)[0]
        ax.axvspan(anom_idx[0], anom_idx[-1], alpha=0.15, color="red")
    ax.set_title(f"Anomaly Score — {model_type}")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "score_timeline.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Score timeline saved: {out_dir}/score_timeline.png")

    # ── 5. Explainability ─────────────────────────────────────
    print("\n[5/6] Computing explanations...")

    # Use a few windows for background
    background = test_windows[:50]  # (50, T, C)

    # SHAP
    shap_exp = SHAPExplainer(model, background, channel_names, device=str(device))
    anomaly_window = test_windows[np.argmax(test_scores)]   # highest-score window
    shap_vals = shap_exp.explain(anomaly_window, n_samples=50)
    top_feats = shap_exp.top_features(shap_vals, k=5)
    print("  Top SHAP features:")
    for name, val in top_feats:
        print(f"    {name:<20}  |SHAP| = {val:.4f}")

    # SHAP bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    top_idx = np.argsort(shap_vals)[::-1][:10]
    ax.barh([channel_names[i] for i in top_idx[::-1]], shap_vals[top_idx[::-1]],
            color=["#e74c3c" if shap_vals[i] > np.median(shap_vals) else "#3498db"
                   for i in top_idx[::-1]])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Attribution (top-10 channels)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "shap_attribution.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  SHAP plot saved: {out_dir}/shap_attribution.png")

    # Causal graph
    causal = CausalGraph(channel_names, max_lag=3)
    causal.fit(data_loader.train_norm)
    root_causes = causal.root_cause(shap_vals, top_k=3)
    report = causal.natural_language_report(shap_vals, float(np.max(test_scores)), top_k=3)
    print("\n" + report)

    with open(os.path.join(out_dir, "anomaly_report.txt"), "w") as f:
        f.write(report)
    print(f"  Report saved: {out_dir}/anomaly_report.txt")

    # Attention (only for Transformer)
    if model_type == "TRANSFORMER":
        attn_viz = AttentionVisualiser(model, device=str(device))
        importance = attn_viz.get_temporal_importance(
            torch.tensor(anomaly_window[None]).to(device)
        )
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.fill_between(range(len(importance)), importance, alpha=0.7, color="#e74c3c")
        ax.set_title("Temporal Attention Weights")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "attention_weights.png"), dpi=120)
        plt.close(fig)
        print(f"  Attention plot saved: {out_dir}/attention_weights.png")

    # ── 6. Digital Twin ───────────────────────────────────────
    print("\n[6/6] Generating Digital Twin images...")

    anom_severity  = min(float(np.max(test_scores)) / calibrator.threshold, 1.0)
    subsystem_probs = shap_vals[:4] / shap_vals[:4].sum()
    subsystem_idx  = int(np.argmax(subsystem_probs))

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i, (sub, ax) in enumerate(zip(
        ["Solar Panel L", "Solar Panel R", "Main Body", "Antenna"], axes
    )):
        is_anom = (i == subsystem_idx)
        img = generate_synthetic_spacecraft_image(
            image_size=128, anomalous=is_anom,
            subsystem=i, severity=anom_severity if is_anom else 0,
        )
        ax.imshow(img)
        ax.set_title(sub + (" ⚠️" if is_anom else ""))
        ax.axis("off")

    plt.suptitle(f"Digital Twin — Anomaly on: {['Solar Panel L','Solar Panel R','Main Body','Antenna'][subsystem_idx]}")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "digital_twin.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Digital Twin saved: {out_dir}/digital_twin.png")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  DEMO COMPLETE")
    print("=" * 65)
    print(f"  Outputs in: {out_dir}/")
    print(f"    score_timeline.png")
    print(f"    shap_attribution.png")
    print(f"    digital_twin.png")
    print(f"    anomaly_report.txt")
    print()
    print("  Next steps:")
    print("    1. Download real data:  SMAP/MSL or OPS-SAT-AD")
    print("    2. python train.py --model GNN --dataset SMAP --epochs 50")
    print("    3. python evaluate.py --checkpoint checkpoints/...")
    print("    4. streamlit run dashboard/app.py")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="GNN",
                        choices=["LSTM_AE", "LSTM_VAE", "TRANSFORMER", "GNN", "FUSION"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out",    default="demo_output")
    args = parser.parse_args()
    run_demo(model_type=args.model, epochs=args.epochs, out_dir=args.out)
