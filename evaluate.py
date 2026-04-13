"""
evaluate.py — Full evaluation of a trained model.

Usage
-----
  python evaluate.py --checkpoint checkpoints/GNN_SMAP_P-1_best.pt
  python evaluate.py --checkpoint checkpoints/TRANSFORMER_OPSSAT_1_best.pt --dataset OPSSAT --channel 1
"""

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve,
)

from train import build_model, get_device
from data.smap_msl import SMAPMSLLoader
from data.opssat   import OpsSatLoader
from data.preprocessing import point_adjust, detection_delay
from detection.detector import ThresholdCalibrator, evaluate, print_metrics
from config import Config


# ──────────────────────────────────────────────────────────────

def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model_type  = ckpt["model_type"]
    n_channels  = ckpt["n_channels"]
    window_size = ckpt["window_size"]

    cfg = Config()
    model = build_model(model_type, n_channels, cfg.model, window_size)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    print(f"Loaded {model_type} checkpoint from epoch {ckpt['epoch']}  "
          f"(loss={ckpt['loss']:.5f}, f1={ckpt.get('f1', '?'):.4f})")
    return model, model_type, n_channels, window_size


def get_scores(model, loader, device):
    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            scores = model.anomaly_score(x).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(y.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


def plot_results(scores, labels, threshold, out_path="eval_results.png"):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    T = np.arange(len(scores))

    # Score timeline
    ax = axes[0]
    ax.plot(T, scores, color="#3498db", linewidth=0.8, label="Anomaly score")
    ax.axhline(threshold, color="#e74c3c", linestyle="--", label=f"Threshold={threshold:.4f}")
    anom_idx = np.where(labels == 1)[0]
    if len(anom_idx):
        ax.axvspan(anom_idx[0], anom_idx[-1], alpha=0.15, color="#e74c3c", label="Ground truth anomaly")
    ax.set_title("Anomaly Score Timeline")
    ax.set_xlabel("Window index")
    ax.legend(fontsize=8)

    # Score distribution
    ax = axes[1]
    ax.hist(scores[labels == 0], bins=80, alpha=0.6, color="#3498db", label="Normal")
    if labels.sum() > 0:
        ax.hist(scores[labels == 1], bins=80, alpha=0.6, color="#e74c3c", label="Anomalous")
    ax.axvline(threshold, color="black", linestyle="--")
    ax.set_title("Score Distribution")
    ax.legend(fontsize=8)

    # PR curve
    ax = axes[2]
    if labels.sum() > 0:
        prec, rec, _ = precision_recall_curve(labels, scores)
        ax.plot(rec, prec, color="#2ecc71")
        ax.set_title("Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No anomalies in test set", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to: {out_path}")
    plt.close(fig)


def run_evaluation(
    checkpoint: str,
    dataset: str = "SMAP",
    channel: str = "P-1",
    batch_size: int = 64,
    out_dir: str = "results",
    device_pref: str = "auto",
):
    device = get_device(device_pref)
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model, model_type, n_channels, window_size = load_checkpoint(checkpoint, device)

    # Load data
    if dataset in ("SMAP", "MSL"):
        data_loader = SMAPMSLLoader(channel=channel, spacecraft=dataset,
                                    window_size=window_size)
    else:
        data_loader = OpsSatLoader(channel_id=channel, window_size=window_size)

    train_dl, test_dl = data_loader.get_loaders(batch_size=batch_size)

    # Calibrate threshold on training scores
    print("\nCalibrating threshold on training data...")
    train_scores, _ = get_scores(model, train_dl, device)
    calibrator = ThresholdCalibrator(method="percentile", percentile=99.0)
    calibrator.fit(train_scores)

    # Evaluate on test set
    print("Evaluating on test set...")
    test_scores, test_labels = get_scores(model, test_dl, device)

    metrics = evaluate(test_scores, test_labels, calibrator.threshold, point_adjust=True)

    tag = f"{model_type}_{dataset}_{channel}"
    print_metrics(metrics, header=f"{tag} — Test Evaluation")

    # Plot
    plot_results(
        test_scores, test_labels, calibrator.threshold,
        out_path=os.path.join(out_dir, f"{tag}_eval.png"),
    )

    # Save scores
    np.save(os.path.join(out_dir, f"{tag}_scores.npy"),  test_scores)
    np.save(os.path.join(out_dir, f"{tag}_labels.npy"),  test_labels)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset",  default="SMAP")
    parser.add_argument("--channel",  default="P-1")
    parser.add_argument("--batch",    type=int, default=64)
    parser.add_argument("--out",      default="results")
    parser.add_argument("--device",   default="auto")
    args = parser.parse_args()

    run_evaluation(
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        channel=args.channel,
        batch_size=args.batch,
        out_dir=args.out,
        device_pref=args.device,
    )
