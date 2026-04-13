"""
explainability/ — Three-layer interpretability for anomaly detections.

  1. SHAPExplainer     — feature attribution (which sensor drove the alert?)
  2. AttentionVisualiser— temporal attribution (which time steps mattered?)
  3. CausalGraph       — sensor dependency graph (root-cause tracing)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import warnings

# ──────────────────────────────────────────────────────────────
# 1. SHAP Explainer
# ──────────────────────────────────────────────────────────────

class SHAPExplainer:
    """
    Wraps a PyTorch anomaly model to compute SHAP values on telemetry windows.
    Uses KernelSHAP (model-agnostic) or DeepSHAP if available.

    Usage
    -----
    exp = SHAPExplainer(model, background_data, channel_names)
    shap_vals = exp.explain(x_window)       # (T, C) → (C,) mean |SHAP|
    exp.plot(shap_vals)
    """

    def __init__(
        self,
        model: nn.Module,
        background: np.ndarray,    # (N_bg, T, C) — normal windows for SHAP baseline
        channel_names: List[str],
        device: str = "cpu",
        n_samples: int = 100,
    ):
        self.model    = model.to(device).eval()
        self.device   = device
        self.names    = channel_names
        self.n_samples = n_samples

        # Build a collapsed background (mean over T) for KernelSHAP
        # shape: (N_bg, T*C) or (N_bg, C) depending on approach
        self.background_3d = background  # (N_bg, T, C)
        self.background_2d = background.mean(axis=1)   # (N_bg, C)

        self._explainer = None

    def _score_fn(self, x_flat: np.ndarray) -> np.ndarray:
        """
        KernelSHAP needs a function that maps (N, C) → (N,).
        We broadcast each row (C,) as a constant across T, then score.
        """
        T = self.background_3d.shape[1]
        x_3d = np.repeat(x_flat[:, None, :], T, axis=1).astype(np.float32)
        t = torch.tensor(x_3d, device=self.device)
        with torch.no_grad():
            scores = self.model.anomaly_score(t).cpu().numpy()
        return scores

    def _get_explainer(self):
        if self._explainer is None:
            try:
                import shap
                self._explainer = shap.KernelExplainer(
                    self._score_fn,
                    self.background_2d[:min(50, len(self.background_2d))],
                )
            except ImportError:
                self._explainer = "fallback"
        return self._explainer

    def explain(self, x: np.ndarray, n_samples: Optional[int] = None) -> np.ndarray:
        """
        x: (T, C) — single window
        Returns: (C,) array of mean absolute SHAP values
        """
        ns = n_samples or self.n_samples
        exp = self._get_explainer()

        if exp == "fallback":
            return self._gradient_fallback(x)

        import shap
        x_2d = x.mean(axis=0, keepdims=True)   # (1, C)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_vals = exp.shap_values(x_2d, nsamples=ns)
        return np.abs(shap_vals[0])   # (C,)

    def _gradient_fallback(self, x: np.ndarray) -> np.ndarray:
        """Integrated-gradients approximation when SHAP is unavailable."""
        T, C = x.shape
        baseline = np.zeros_like(x)
        steps = 20
        grads = np.zeros(C)
        for alpha in np.linspace(0, 1, steps):
            interp = torch.tensor(
                (baseline + alpha * (x - baseline))[None].astype(np.float32),
                requires_grad=True, device=self.device
            )
            score = self.model.anomaly_score(interp).sum()
            score.backward()
            grads += interp.grad.cpu().numpy()[0].mean(axis=0)
        return np.abs(grads * (x - baseline).mean(axis=0) / steps)

    def top_features(self, shap_vals: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Return top-k (channel_name, importance) pairs."""
        idx = np.argsort(shap_vals)[::-1][:k]
        return [(self.names[i], float(shap_vals[i])) for i in idx]

    def plot(self, shap_vals: np.ndarray, title: str = "Feature Attribution") -> None:
        import matplotlib.pyplot as plt
        idx = np.argsort(shap_vals)[::-1]
        fig, ax = plt.subplots(figsize=(10, 4))
        names = [self.names[i] for i in idx]
        vals  = shap_vals[idx]
        colors = ["#e74c3c" if v > np.median(vals) else "#3498db" for v in vals]
        ax.barh(names[::-1], vals[::-1], color=colors[::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
        return fig


# ──────────────────────────────────────────────────────────────
# 2. Attention Visualiser
# ──────────────────────────────────────────────────────────────

class AttentionVisualiser:
    """
    Extracts and plots attention weights from a Transformer-based model.
    Works with TransformerAD which exposes get_attention_weights().
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model  = model.to(device).eval()
        self.device = device

    def get_weights(self, x: torch.Tensor) -> Optional[np.ndarray]:
        """
        x: (1, T, C)
        Returns: (T, T) — averaged across heads, or None if model lacks attention.
        """
        if not hasattr(self.model, "get_attention_weights"):
            return None
        x = x.to(self.device)
        weights = self.model.get_attention_weights(x)   # (1, heads, T, T)
        return weights[0].mean(0).cpu().numpy()         # (T, T)

    def get_temporal_importance(self, x: torch.Tensor) -> np.ndarray:
        """
        Returns (T,) importance per time step — row-sum of attn matrix.
        """
        w = self.get_weights(x)
        if w is None:
            # Fallback: gradient-based saliency
            return self._saliency_fallback(x)
        return w.sum(axis=0) / w.sum()

    def _saliency_fallback(self, x: torch.Tensor) -> np.ndarray:
        x = x.to(self.device).requires_grad_(True)
        score = self.model.anomaly_score(x).sum()
        score.backward()
        return x.grad.abs().mean(dim=(0, 2)).cpu().numpy()

    def plot(
        self,
        x: torch.Tensor,
        time_labels: Optional[List[str]] = None,
        title: str = "Temporal Attention",
    ):
        import matplotlib.pyplot as plt
        importance = self.get_temporal_importance(x)
        T = len(importance)
        t = time_labels or list(range(T))

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.fill_between(range(T), importance, alpha=0.6, color="#e74c3c")
        ax.plot(range(T), importance, color="#c0392b", linewidth=1)
        ax.set_ylabel("Attention weight")
        ax.set_title(title)
        ax.set_xticks(range(0, T, max(1, T // 10)))
        plt.tight_layout()
        plt.show()
        return fig


# ──────────────────────────────────────────────────────────────
# 3. Causal Graph
# ──────────────────────────────────────────────────────────────

class CausalGraph:
    """
    Builds a Granger-causality sensor dependency graph and uses it to
    trace the most likely root cause of a detected anomaly.

    Usage
    -----
    cg = CausalGraph(channel_names)
    cg.fit(train_data)                       # (T, C) normal data
    root = cg.root_cause(shap_vals, top_k=3)
    cg.plot()
    """

    def __init__(
        self,
        channel_names: List[str],
        max_lag: int = 5,
        significance: float = 0.05,
    ):
        self.names = channel_names
        self.max_lag = max_lag
        self.significance = significance
        self.adj: Optional[np.ndarray] = None       # (C, C) causal strengths
        self._graph = None

    def fit(self, data: np.ndarray):
        """
        data: (T, C)
        Computes a simplified Granger-causality adjacency matrix.
        For each pair (i, j): does channel i Granger-cause channel j?
        """
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
        except ImportError:
            print("[CausalGraph] statsmodels not installed — using correlation proxy.")
            self.adj = np.abs(np.corrcoef(data.T))
            np.fill_diagonal(self.adj, 0)
            self._build_graph()
            return

        C = data.shape[1]
        self.adj = np.zeros((C, C))
        for i in range(C):
            for j in range(C):
                if i == j:
                    continue
                try:
                    pair = np.column_stack([data[:, j], data[:, i]])
                    res = grangercausalitytests(pair, self.max_lag, verbose=False)
                    p_vals = [res[lag][0]["ssr_ftest"][1] for lag in range(1, self.max_lag + 1)]
                    min_p  = min(p_vals)
                    if min_p < self.significance:
                        self.adj[i, j] = 1.0 - min_p   # strength ~ (1 - p)
                except Exception:
                    pass
        self._build_graph()

    def _build_graph(self):
        try:
            import networkx as nx
            C = len(self.names)
            G = nx.DiGraph()
            G.add_nodes_from(range(C))
            for i in range(C):
                for j in range(C):
                    if self.adj[i, j] > 0:
                        G.add_edge(i, j, weight=float(self.adj[i, j]))
            self._graph = G
        except ImportError:
            pass

    def root_cause(
        self,
        shap_vals: np.ndarray,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Given SHAP importances per channel, trace upstream causes through the
        causal graph (BFS traversal backwards from the most anomalous sensor).
        Returns a list of {node, name, role, strength} dicts.
        """
        if self.adj is None:
            return [{"name": self.names[i], "role": "unknown"}
                    for i in np.argsort(shap_vals)[::-1][:top_k]]

        # Start from the top SHAP node
        target = int(np.argmax(shap_vals))
        results = [{"node": target, "name": self.names[target],
                    "role": "symptom", "strength": float(shap_vals[target])}]

        # Walk backwards: which nodes Granger-cause the target?
        causes = np.argsort(self.adj[:, target])[::-1]
        for c in causes[:top_k - 1]:
            if self.adj[c, target] > 0:
                results.append({
                    "node": int(c),
                    "name": self.names[c],
                    "role": "cause",
                    "strength": float(self.adj[c, target]),
                })

        return results

    def natural_language_report(
        self,
        shap_vals: np.ndarray,
        anomaly_score: float,
        top_k: int = 3,
    ) -> str:
        """Generate a human-readable alert report."""
        roots = self.root_cause(shap_vals, top_k)
        symptom = next((r for r in roots if r["role"] == "symptom"), roots[0])
        causes  = [r for r in roots if r["role"] == "cause"]

        lines = [
            f"⚠️  ANOMALY DETECTED  (score = {anomaly_score:.4f})",
            "",
            f"Primary symptom : {symptom['name']} "
            f"(attribution = {symptom['strength']:.3f})",
        ]
        if causes:
            cause_str = ", ".join(f"{c['name']} (strength={c['strength']:.2f})"
                                  for c in causes)
            lines.append(f"Likely root cause(s): {cause_str}")
        else:
            lines.append("Root cause: insufficient causal data — manual inspection advised.")

        lines += [
            "",
            "Recommended actions:",
            "  1. Inspect telemetry for " + symptom["name"],
            "  2. Cross-check with subsystem logs",
            "  3. Alert ground operations team",
        ]
        return "\n".join(lines)

    def plot(self, highlight: Optional[List[int]] = None):
        """Plot the causal graph with anomalous nodes highlighted."""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError:
            print("[CausalGraph] networkx / matplotlib not installed.")
            return

        if self._graph is None:
            print("[CausalGraph] Graph not built yet — call .fit() first.")
            return

        G = self._graph
        pos = nx.spring_layout(G, seed=42)
        colors = []
        for n in G.nodes():
            if highlight and n in highlight:
                colors.append("#e74c3c")
            else:
                colors.append("#3498db")

        fig, ax = plt.subplots(figsize=(10, 7))
        nx.draw_networkx(
            G, pos=pos, ax=ax,
            node_color=colors,
            labels={i: self.names[i] for i in G.nodes()},
            arrows=True,
            edge_color="#aaa",
            node_size=800,
            font_size=8,
        )
        ax.set_title("Sensor Causal Dependency Graph")
        plt.tight_layout()
        plt.show()
        return fig
