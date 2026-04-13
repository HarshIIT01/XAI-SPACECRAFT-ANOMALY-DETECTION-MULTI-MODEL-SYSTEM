"""
models/graph_rnn.py — Dynamic Graph Neural Network + GRU for telemetry AD.

Inspired by STGLR (2025):
  1. Build an inter-sensor graph (static from correlation OR learned dynamically)
  2. GraphSAGE message-passing to fuse neighbouring sensor features
  3. GRU temporal encoding
  4. VAE bottleneck for reconstruction-based scoring

Requires: torch-geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

try:
    from torch_geometric.nn import SAGEConv, GATConv, GCNConv
    _TG_AVAILABLE = True
except ImportError:
    _TG_AVAILABLE = False
    print("[GraphRNN] torch-geometric not installed — using fallback adjacency MLP.")


# ──────────────────────────────────────────────────────────────
# Fallback: simple adjacency-weighted MLP (no torch-geometric)
# ──────────────────────────────────────────────────────────────

class AdjMLP(nn.Module):
    """
    Simple graph convolution without torch-geometric:
      h_out = W * (A_norm @ h_in)
    """
    def __init__(self, in_dim, out_dim, n_nodes):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        # learnable adjacency
        self.adj = nn.Parameter(torch.eye(n_nodes) +
                                torch.randn(n_nodes, n_nodes) * 0.01)

    def forward(self, x, edge_index=None, edge_weight=None):
        # x: (B*T, C, F)  — node features
        A = torch.softmax(self.adj, dim=-1)          # row-normalise
        # x shape: (batch, n_nodes, features)
        h = torch.bmm(A.unsqueeze(0).expand(x.size(0), -1, -1), x)
        return F.relu(self.fc(h))


# ──────────────────────────────────────────────────────────────
# Dynamic Graph Constructor
# ──────────────────────────────────────────────────────────────

class DynamicGraphConstructor(nn.Module):
    """
    Learns a dynamic adjacency matrix from current node embeddings.
    Uses a gating mechanism similar to STGLR's adaptive graph.
    """

    def __init__(self, n_nodes: int, embed_dim: int = 16, threshold: float = 0.0):
        super().__init__()
        self.n_nodes = n_nodes
        self.threshold = threshold
        self.node_emb1 = nn.Embedding(n_nodes, embed_dim)
        self.node_emb2 = nn.Embedding(n_nodes, embed_dim)
        self.lin1 = nn.Linear(embed_dim, embed_dim)
        self.lin2 = nn.Linear(embed_dim, embed_dim)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (edge_index, edge_weight) for a fully learnable sparse graph."""
        idx = torch.arange(self.n_nodes, device=self.node_emb1.weight.device)
        e1 = torch.tanh(self.lin1(self.node_emb1(idx)))   # (C, D)
        e2 = torch.tanh(self.lin2(self.node_emb2(idx)))   # (C, D)
        A = F.relu(torch.mm(e1, e2.T) - torch.mm(e2, e1.T))  # antisym
        A = torch.softmax(A, dim=1)

        # Sparsify
        mask = A > self.threshold
        rows, cols = mask.nonzero(as_tuple=True)
        edge_index  = torch.stack([rows, cols], dim=0)     # (2, E)
        edge_weight = A[rows, cols]                        # (E,)
        return edge_index, edge_weight


# ──────────────────────────────────────────────────────────────
# Graph + GRU Encoder
# ──────────────────────────────────────────────────────────────

class GraphGRUEncoder(nn.Module):
    """
    Processes a multivariate window (B, T, C):
      Step 1 — for each time step, run GNN over C sensor nodes
      Step 2 — run GRU over the resulting node-level time series
      Step 3 — pool sensor features → latent z
    """

    def __init__(
        self,
        n_channels: int,
        node_feat_dim: int,        # = 1 (raw value) or more
        gnn_dim: int = 32,
        gru_dim: int = 64,
        latent_dim: int = 32,
        gnn_type: str = "SAGE",
        num_gnn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_channels  = n_channels
        self.gnn_dim     = gnn_dim
        self.gru_dim     = gru_dim
        self.latent_dim  = latent_dim

        # Dynamic graph
        self.graph_ctor = DynamicGraphConstructor(n_channels, embed_dim=16)

        # GNN layers
        if _TG_AVAILABLE:
            gnn_cls = {"SAGE": SAGEConv, "GAT": GATConv, "GCN": GCNConv}[gnn_type]
            self.gnn_layers = nn.ModuleList([
                gnn_cls(node_feat_dim if i == 0 else gnn_dim, gnn_dim)
                for i in range(num_gnn_layers)
            ])
        else:
            self.gnn_layers = nn.ModuleList([
                AdjMLP(node_feat_dim if i == 0 else gnn_dim, gnn_dim, n_channels)
                for i in range(num_gnn_layers)
            ])

        self.gnn_norm = nn.LayerNorm(gnn_dim)

        # GRU for temporal modelling (per-node)
        self.gru = nn.GRU(
            input_size=gnn_dim,
            hidden_size=gru_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        # Pool + project to latent
        self.pool_fc  = nn.Linear(gru_dim * 2, latent_dim)
        self.mu_fc    = nn.Linear(latent_dim, latent_dim)
        self.logv_fc  = nn.Linear(latent_dim, latent_dim)
        self.dropout  = nn.Dropout(dropout)

    def _gnn_step(
        self,
        node_feat: torch.Tensor,   # (B, C, fd)
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """One GNN pass over all batch items simultaneously."""
        B, C, fd = node_feat.shape
        # flatten batch: (B*C, fd)
        x = node_feat.reshape(B * C, fd)

        # shift edge_index for each batch item
        batch_edge_lists = []
        for b in range(B):
            shifted = edge_index + b * C
            batch_edge_lists.append(shifted)
        batch_ei = torch.cat(batch_edge_lists, dim=1)  # (2, B*E)

        # GNN forward
        if _TG_AVAILABLE:
            for layer in self.gnn_layers:
                try:
                    x = F.relu(layer(x, batch_ei))
                except TypeError:
                    x = F.relu(layer(x, batch_ei, edge_weight=None))
        else:
            x_3d = x.reshape(B, C, -1)
            for layer in self.gnn_layers:
                x_3d = layer(x_3d)
            x = x_3d.reshape(B * C, -1)  # flatten back

        return self.gnn_norm(x.reshape(B, C, -1))  # (B, C, gnn_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, C)
        Returns: mu (B, latent_dim), log_var (B, latent_dim)
        """
        B, T, C = x.shape
        edge_index, edge_weight = self.graph_ctor()

        # Per-step GNN: process each time step
        gnn_out = []
        for t in range(T):
            feat = x[:, t, :].unsqueeze(-1)          # (B, C, 1)
            h = self._gnn_step(feat, edge_index, edge_weight)   # (B, C, gnn_dim)
            gnn_out.append(h)
        gnn_seq = torch.stack(gnn_out, dim=2)        # (B, C, T, gnn_dim)

        # GRU per node, pool across nodes
        gnn_seq_flat = gnn_seq.reshape(B * C, T, self.gnn_dim)
        gru_out, _ = self.gru(gnn_seq_flat)           # (B*C, T, 2*gru)
        gru_out = gru_out[:, -1, :]                   # (B*C, 2*gru_dim)
        gru_out = gru_out.reshape(B, C, -1)           # (B, C, 2*gru_dim)
        pooled  = gru_out.mean(dim=1)                 # (B, 2*gru_dim)  — mean pool

        h = F.relu(self.pool_fc(self.dropout(pooled)))  # (B, latent_dim)
        return self.mu_fc(h), self.logv_fc(h)


# ──────────────────────────────────────────────────────────────
# Full GNN-VAE model
# ──────────────────────────────────────────────────────────────

class GraphVAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_channels, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, 2, batch_first=True)
        self.out  = nn.Linear(hidden_dim, n_channels)

    def forward(self, z):
        h = F.relu(self.fc(z)).unsqueeze(1).expand(-1, self.seq_len, -1)
        out, _ = self.lstm(h)
        return self.out(out)   # (B, T, C)


class GNNVariationalAD(nn.Module):
    """STGLR-inspired Graph + GRU + VAE anomaly detector."""

    def __init__(
        self,
        n_channels: int = 25,
        gnn_dim: int = 32,
        gru_dim: int = 64,
        latent_dim: int = 32,
        seq_len: int = 128,
        gnn_type: str = "SAGE",
        beta: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = GraphGRUEncoder(
            n_channels, 1, gnn_dim, gru_dim, latent_dim, gnn_type, dropout=dropout
        )
        self.decoder = GraphVAEDecoder(latent_dim, gru_dim, n_channels, seq_len)
        self.beta = beta
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        if self.training:
            return mu + (0.5 * log_var).exp() * torch.randn_like(mu)
        return mu

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    def loss(self, x):
        x_hat, mu, log_var = self.forward(x)
        recon = F.mse_loss(x_hat, x)
        kl    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon + self.beta * kl, recon, kl

    def anomaly_score(self, x):
        with torch.no_grad():
            x_hat, mu, log_var = self.forward(x)
            recon = ((x - x_hat) ** 2).mean(dim=(1, 2))
            kl    = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1)
            return recon + 0.1 * kl

    def get_graph_adjacency(self) -> torch.Tensor:
        """Returns the current learned adjacency matrix for visualisation."""
        with torch.no_grad():
            idx = torch.arange(self.encoder.n_channels,
                               device=next(self.parameters()).device)
            e1 = torch.tanh(self.encoder.graph_ctor.lin1(
                self.encoder.graph_ctor.node_emb1(idx)))
            e2 = torch.tanh(self.encoder.graph_ctor.lin2(
                self.encoder.graph_ctor.node_emb2(idx)))
            A = F.relu(torch.mm(e1, e2.T) - torch.mm(e2, e1.T))
            return torch.softmax(A, dim=1)