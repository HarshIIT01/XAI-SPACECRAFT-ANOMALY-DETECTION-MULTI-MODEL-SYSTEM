"""
detection/detector.py — Core anomaly detector with threshold calibration,
                         dual-stage pipeline, and streaming inference.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
from collections import deque
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import mahalanobis


# ──────────────────────────────────────────────────────────────
# Threshold calibrator
# ──────────────────────────────────────────────────────────────

class ThresholdCalibrator:
    """
    Fits an anomaly threshold on validation/training reconstruction errors.
    Supports percentile, mean+std, and ROC-optimal methods.
    """

    def __init__(self, method: str = "percentile", percentile: float = 99.0, std_factor: float = 3.0):
        self.method = method
        self.percentile = percentile
        self.std_factor = std_factor
        self.threshold: Optional[float] = None

    def fit(self, scores: np.ndarray) -> "ThresholdCalibrator":
        if self.method == "percentile":
            self.threshold = float(np.percentile(scores, self.percentile))
        elif self.method == "std":
            self.threshold = float(scores.mean() + self.std_factor * scores.std())
        elif self.method == "fixed":
            self.threshold = self.threshold or float(np.percentile(scores, 95))
        else:
            self.threshold = float(np.percentile(scores, self.percentile))
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        assert self.threshold is not None, "Call .fit() first."
        return (scores > self.threshold).astype(int)

    def roc_optimal(
        self, scores: np.ndarray, labels: np.ndarray
    ) -> "ThresholdCalibrator":
        """Find threshold that maximises F1 on labelled data."""
        from sklearn.metrics import f1_score
        best_f1, best_thr = 0.0, 0.0
        for thr in np.percentile(scores, np.linspace(80, 100, 50)):
            preds = (scores > thr).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        self.threshold = float(best_thr)
        return self


# ──────────────────────────────────────────────────────────────
# Fast (lightweight) detectors — stage 1
# ──────────────────────────────────────────────────────────────

class MahalanobisDetector:
    """
    Classical Mahalanobis-distance anomaly detector.
    Fits on normal training data and scores new windows by their
    distance from the normal distribution in feature space.
    """

    def __init__(self):
        self.cov_estimator = EmpiricalCovariance()
        self.mean_: Optional[np.ndarray] = None
        self.cov_inv_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MahalanobisDetector":
        """X: (N, F) — feature vectors of normal windows."""
        self.cov_estimator.fit(X)
        self.mean_    = self.cov_estimator.location_
        try:
            self.cov_inv_ = np.linalg.inv(self.cov_estimator.covariance_)
        except np.linalg.LinAlgError:
            self.cov_inv_ = np.linalg.pinv(self.cov_estimator.covariance_)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """X: (N, F) → (N,) Mahalanobis distances."""
        assert self.mean_ is not None
        dists = np.array([
            mahalanobis(x, self.mean_, self.cov_inv_) for x in X
        ])
        return dists

    def anomaly_score(self, x: torch.Tensor) -> np.ndarray:
        """Convenience: accept a (B, T, C) tensor → flatten → score."""
        X = x.cpu().numpy().reshape(x.shape[0], -1)
        return self.score(X)


class IsolationForestDetector:
    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
        )

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(X.reshape(len(X), -1))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        flat = X.reshape(len(X), -1)
        return -self.model.score_samples(flat)   # higher = more anomalous

    def anomaly_score(self, x: torch.Tensor) -> np.ndarray:
        return self.score(x.cpu().numpy())


# ──────────────────────────────────────────────────────────────
# Deep detector — stage 2
# ──────────────────────────────────────────────────────────────

class DeepDetector:
    """
    Wraps any PyTorch model that exposes .anomaly_score(x) → Tensor.
    Handles device management and batched inference.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        batch_size: int = 64,
    ):
        self.model      = model.to(device).eval()
        self.device     = device
        self.batch_size = batch_size
        self.calibrator = ThresholdCalibrator()

    def calibrate(self, train_loader) -> "DeepDetector":
        """Run model on training data to set the anomaly threshold."""
        all_scores = []
        with torch.no_grad():
            for x, _ in train_loader:
                x = x.to(self.device)
                s = self.model.anomaly_score(x).cpu().numpy()
                all_scores.append(s)
        all_scores = np.concatenate(all_scores)
        self.calibrator.fit(all_scores)
        print(f"[DeepDetector] Threshold calibrated: {self.calibrator.threshold:.6f}")
        return self

    def score_batch(self, x: torch.Tensor) -> np.ndarray:
        x = x.to(self.device)
        with torch.no_grad():
            return self.model.anomaly_score(x).cpu().numpy()

    def predict_batch(self, x: torch.Tensor) -> np.ndarray:
        scores = self.score_batch(x)
        return self.calibrator.predict(scores)

    def score_array(self, X: np.ndarray) -> np.ndarray:
        """X: (N, T, C) numpy → (N,) scores."""
        all_scores = []
        for start in range(0, len(X), self.batch_size):
            batch = torch.tensor(X[start:start + self.batch_size], dtype=torch.float32)
            all_scores.append(self.score_batch(batch))
        return np.concatenate(all_scores)


# ──────────────────────────────────────────────────────────────
# Dual-Stage Pipeline
# ──────────────────────────────────────────────────────────────

class DualStagePipeline:
    """
    Stage 1 (fast)  : lightweight detector (Mahalanobis or IsoForest) — real-time
    Stage 2 (deep)  : deep model (LSTM-AE / Transformer / GNN) — detailed analysis

    Only triggers Stage 2 when Stage 1 raises an alert, saving compute.
    Smooths alerts over a sliding window to reduce false positives.
    """

    def __init__(
        self,
        fast_detector,               # MahalanobisDetector or IsolationForestDetector
        deep_detector: DeepDetector,
        alert_window: int = 10,
        alert_threshold_ratio: float = 0.5,   # majority vote ratio
    ):
        self.fast   = fast_detector
        self.deep   = deep_detector
        self.alert_window = alert_window
        self.alert_threshold_ratio = alert_threshold_ratio
        self._alert_buffer = deque(maxlen=alert_window)

    def step(
        self,
        x: np.ndarray,     # single window (T, C)
        run_deep: bool = False,
    ) -> Dict:
        """
        Process one window in streaming fashion.
        Returns dict with alert, scores, and stage used.
        """
        x_2d = x.reshape(1, -1)
        fast_score = float(self.fast.score(x_2d)[0])
        fast_alert = fast_score > self.fast_threshold

        result = {
            "fast_score" : fast_score,
            "fast_alert" : fast_alert,
            "deep_score" : None,
            "deep_alert" : None,
            "final_alert": False,
            "stage"      : 1,
        }

        if fast_alert or run_deep:
            x_t = torch.tensor(x[None], dtype=torch.float32)
            deep_score = float(self.deep.score_batch(x_t)[0])
            deep_alert = deep_score > self.deep.calibrator.threshold
            result.update({
                "deep_score" : deep_score,
                "deep_alert" : deep_alert,
                "stage"      : 2,
            })
            self._alert_buffer.append(int(deep_alert))
        else:
            self._alert_buffer.append(int(fast_alert))

        # Majority-vote smoothing
        if len(self._alert_buffer) == self.alert_window:
            ratio = sum(self._alert_buffer) / self.alert_window
            result["final_alert"] = ratio >= self.alert_threshold_ratio

        return result

    def set_fast_threshold(self, train_X: np.ndarray, percentile: float = 99.0):
        scores = self.fast.score(train_X.reshape(len(train_X), -1))
        self.fast_threshold = float(np.percentile(scores, percentile))
        print(f"[DualStage] Fast threshold: {self.fast_threshold:.4f}")


# ──────────────────────────────────────────────────────────────
# Evaluation utilities
# ──────────────────────────────────────────────────────────────

def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    point_adjust: bool = True,
) -> Dict[str, float]:
    """
    Full evaluation report.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    from data.preprocessing import point_adjust as _pa, detection_delay

    preds = (scores > threshold).astype(int)

    if point_adjust:
        preds = _pa(labels, preds)

    metrics = {
        "precision"       : precision_score(labels, preds, zero_division=0),
        "recall"          : recall_score(labels, preds, zero_division=0),
        "f1"              : f1_score(labels, preds, zero_division=0),
        "roc_auc"         : roc_auc_score(labels, scores) if labels.sum() > 0 else float("nan"),
        "detection_delay" : detection_delay(labels, preds),
        "threshold"       : threshold,
    }
    return metrics


def print_metrics(metrics: Dict[str, float], header: str = "Evaluation Results"):
    print(f"\n{'='*40}")
    print(f" {header}")
    print(f"{'='*40}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")
    print(f"{'='*40}\n")
