"""
End-to-end demo pipeline (synthetic data if real files are missing).

Usage (from project root)::

  python scripts/run_pipeline.py --model GNN --epochs 5
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW

from _bootstrap import bootstrap

bootstrap()

from spacecraft_anomaly.config import Config  # noqa: E402
from spacecraft_anomaly.data.smap_msl import SMAPMSLLoader  # noqa: E402
from spacecraft_anomaly.detection.detector import (  # noqa: E402
    ThresholdCalibrator,
    evaluate,
    print_metrics,
)
from spacecraft_anomaly.explainability import (  # noqa: E402
    AttentionVisualiser,
    CausalGraph,
    SHAPExplainer,
)
from spacecraft_anomaly.models.digital_twin import generate_synthetic_spacecraft_image  # noqa: E402
from spacecraft_anomaly.paths import resolve_path  # noqa: E402
from spacecraft_anomaly.training import build_model, get_device, train_epoch  # noqa: E402


def run_demo(model_type: str = "GNN", epochs: int = 5, out_dir: str = "demo_output"):
    out_dir = str(resolve_path(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    cfg = Config()
    device = get_device("auto")

    print("=" * 65)
    print("  SPACECRAFT ANOMALY DETECTION — FULL PIPELINE DEMO")
    print("=" * 65)

    print("\n[1/6] Loading data (synthetic placeholder if files missing)...")
    data_loader = SMAPMSLLoader(window_size=128)
    data_loader.summary()
    train_dl, test_dl = data_loader.get_loaders(batch_size=32)
    n_channels = data_loader.n_channels
    channel_names = data_loader.channel_names

    print(f"\n[2/6] Building {model_type} model...")
    model = build_model(model_type, n_channels, cfg.model, seq_len=128).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=1e-3)

    print(f"\n[3/6] Training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_dl, optimizer, device, model_type, epoch, epochs)
        print(f"  Epoch {epoch}/{epochs}  |  loss={loss:.5f}")

    print("\n[4/6] Evaluating on test set...")
    model.eval()

    train_scores = []
    with torch.no_grad():
        for x, _ in train_dl:
            s = model.anomaly_score(x.to(device)).cpu().numpy()
            train_scores.append(s)
    train_scores = np.concatenate(train_scores)

    calibrator = ThresholdCalibrator("percentile", 99.0)
    calibrator.fit(train_scores)

    test_scores, test_labels, test_windows = [], [], []
    with torch.no_grad():
        for x, y in test_dl:
            s = model.anomaly_score(x.to(device)).cpu().numpy()
            test_scores.append(s)
            test_labels.append(y.numpy())
            test_windows.append(x.cpu().numpy())

    test_scores = np.concatenate(test_scores)
    test_labels = np.concatenate(test_labels)
    test_windows = np.concatenate(test_windows)

    metrics = evaluate(test_scores, test_labels, calibrator.threshold)
    print_metrics(metrics, f"{model_type} — Evaluation")

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

    print("\n[5/6] Computing explanations...")
    background = test_windows[:50]
    # SHAP batches many windows; use CPU to avoid GPU OOM on small cards
    explain_device = "cpu" if device.type == "cuda" else str(device)
    shap_exp = SHAPExplainer(model, background, channel_names, device=explain_device)
    anomaly_window = test_windows[np.argmax(test_scores)]
    shap_vals = shap_exp.explain(anomaly_window, n_samples=50)
    top_feats = shap_exp.top_features(shap_vals, k=5)
    print("  Top SHAP features:")
    for name, val in top_feats:
        print(f"    {name:<20}  |SHAP| = {val:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    top_idx = np.argsort(shap_vals)[::-1][:10]
    ax.barh(
        [channel_names[i] for i in top_idx[::-1]],
        shap_vals[top_idx[::-1]],
        color=[
            "#e74c3c" if shap_vals[i] > np.median(shap_vals) else "#3498db"
            for i in top_idx[::-1]
        ],
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Attribution (top-10 channels)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "shap_attribution.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  SHAP plot saved: {out_dir}/shap_attribution.png")

    causal = CausalGraph(channel_names, max_lag=3)
    causal.fit(data_loader.train_norm)
    report = causal.natural_language_report(shap_vals, float(np.max(test_scores)), top_k=3)
    print("\n" + report)

    with open(os.path.join(out_dir, "anomaly_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report saved: {out_dir}/anomaly_report.txt")

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

    print("\n[6/6] Generating Digital Twin images...")
    anom_severity = min(float(np.max(test_scores)) / calibrator.threshold, 1.0)
    subsystem_probs = shap_vals[:4] / shap_vals[:4].sum()
    subsystem_idx = int(np.argmax(subsystem_probs))
    subsystems = ["Solar Panel L", "Solar Panel R", "Main Body", "Antenna"]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i, (sub, ax) in enumerate(zip(subsystems, axes)):
        is_anom = i == subsystem_idx
        img = generate_synthetic_spacecraft_image(
            image_size=128,
            anomalous=is_anom,
            subsystem=i,
            severity=anom_severity if is_anom else 0,
        )
        ax.imshow(img)
        ax.set_title(sub + (" [!]" if is_anom else ""))
        ax.axis("off")

    plt.suptitle(f"Digital Twin — Anomaly on: {subsystems[subsystem_idx]}")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "digital_twin.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Digital Twin saved: {out_dir}/digital_twin.png")

    print("\n" + "=" * 65)
    print("  DEMO COMPLETE")
    print("=" * 65)
    print(f"  Outputs in: {out_dir}/")
    print("  Next steps:")
    print("    1. Download real data: SMAP/MSL or OPS-SAT-AD")
    print("    2. python scripts/train.py --model GNN --dataset SMAP --epochs 50")
    print("    3. python scripts/evaluate.py --checkpoint checkpoints/...")
    print("    4. Launch the web app: uvicorn webapp.main:app --reload")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="GNN",
        choices=["LSTM_AE", "LSTM_VAE", "TRANSFORMER", "GNN", "FUSION"],
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", default="demo_output")
    args = parser.parse_args()
    run_demo(model_type=args.model, epochs=args.epochs, out_dir=args.out)


if __name__ == "__main__":
    main()
