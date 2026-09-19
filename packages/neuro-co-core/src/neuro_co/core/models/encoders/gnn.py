"""Sparse graph encoder built with PyTorch Geometric."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn

try:
    from torch_geometric.nn import TransformerConv
except ImportError as exc:
    raise ImportError("Install neuro-co-core[gnn] to use the GNN encoder.") from exc

from ..encoder import pool_mean


class RBFDistanceEncoding(nn.Module):
    """Encode normalized edge distance as d, d², RBFs, and Fourier pairs."""

    centers: Tensor
    gamma: Tensor
    fourier_frequency: Tensor

    def __init__(self, rbf_k: int = 16, fourier_feats: int = 1) -> None:
        super().__init__()
        if rbf_k < 1:
            raise ValueError("rbf_k must be positive")
        if fourier_feats < 0:
            raise ValueError("fourier_feats cannot be negative")
        centers = torch.linspace(0.0, 1.0, rbf_k)
        delta = centers[1] - centers[0] if rbf_k > 1 else torch.tensor(1.0)
        gamma = torch.full_like(centers, 1.0 / (2.0 * float(delta) ** 2))
        self.register_buffer("centers", centers)
        self.register_buffer("gamma", gamma)
        self.register_buffer("fourier_frequency", torch.randn(fourier_feats) * 2.0 * math.pi)
        self.rbf_k = rbf_k
        self.fourier_feats = fourier_feats

    @property
    def output_dim(self) -> int:
        return 2 + self.rbf_k + 2 * self.fourier_feats

    def forward(self, distance: Float[Tensor, "b e"]) -> Float[Tensor, "b e f"]:
        squared = distance.square()
        rbf = torch.exp(
            -self.gamma.view(1, 1, -1)
            * (distance.unsqueeze(-1) - self.centers.view(1, 1, -1)).square()
        )
        features = [distance.unsqueeze(-1), squared.unsqueeze(-1), rbf]
        if self.fourier_feats:
            phase = distance.unsqueeze(-1) * self.fourier_frequency.view(1, 1, -1)
            features.extend((torch.sin(phase), torch.cos(phase)))
        return torch.cat(features, dim=-1)


class _BatchNormLast(nn.Module):
    """BatchNorm over the final feature dimension for rank-2 or rank-3 input."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.normalizer = nn.BatchNorm1d(hidden_dim)

    def forward(self, value: Tensor) -> Tensor:
        shape = value.shape
        return self.normalizer(value.reshape(-1, shape[-1])).reshape(shape)


class _GNNBlock(nn.Module):
    """Pre-normalized sparse message attention followed by a residual FFN."""

    def __init__(self, hidden_dim: int = 128, num_heads: int = 8) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.norm_gnn = _BatchNormLast(hidden_dim)
        self.norm_ffn = _BatchNormLast(hidden_dim)
        self.gnn = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            edge_dim=hidden_dim,
            bias=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(
        self,
        node_features: Float[Tensor, "b n d"],
        source: Int[Tensor, "b e"],
        destination: Int[Tensor, "b e"],
        edge_features: Float[Tensor, "b e d"],
    ) -> Float[Tensor, "b n d"]:
        batch, nodes, hidden_dim = node_features.shape
        edges = source.shape[1]
        offsets = torch.arange(batch, device=source.device).unsqueeze(1) * nodes
        edge_index = torch.stack(
            ((source + offsets).reshape(-1), (destination + offsets).reshape(-1)), dim=0
        )
        normalized = self.norm_gnn(node_features).reshape(batch * nodes, hidden_dim)
        message = self.gnn(normalized, edge_index, edge_features.reshape(batch * edges, -1))
        hidden = node_features + message.reshape(batch, nodes, hidden_dim)
        return hidden + self.ffn(self.norm_ffn(hidden))


class GNNEncoder(nn.Module):
    """Sparse geometric GNN satisfying MLCO's encoder protocol."""

    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        *,
        k_sparse: int = 10,
        dropout: float = 0.1,
        rbf_k: int = 16,
        fourier_feats: int = 1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if in_dim < 2:
            raise ValueError("GNN features must begin with x and y coordinates")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if k_sparse < 1:
            raise ValueError("k_sparse must be positive")
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.k_sparse = k_sparse
        self.dropout = dropout
        self.rbf_k = rbf_k
        self.fourier_feats = fourier_feats
        self.residual = residual
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.distance_encoding = RBFDistanceEncoding(rbf_k, fourier_feats)
        edge_dim = self.distance_encoding.output_dim + 2
        self.edge_projection = nn.Linear(edge_dim, hidden_dim)
        self.blocks = nn.ModuleList([_GNNBlock(hidden_dim, num_heads) for _ in range(num_layers)])

    def sparse_edge_index(
        self, coordinates: Float[Tensor, "b n 2"]
    ) -> tuple[Int[Tensor, "b e"], Int[Tensor, "b e"]]:
        """Return source and destination indices for kNN plus depot edges."""

        if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
            raise ValueError("coordinates must have shape (batch, nodes, 2)")
        batch, nodes, _ = coordinates.shape
        if nodes < 2:
            raise ValueError("GNN requires a depot and at least one customer")
        neighbours = min(self.k_sparse, nodes - 1)
        distance = torch.cdist(coordinates, coordinates)
        diagonal = torch.eye(nodes, dtype=torch.bool, device=coordinates.device).unsqueeze(0)
        distance = distance.masked_fill(diagonal, torch.inf)
        nearest = distance.topk(neighbours, dim=-1, largest=False, sorted=True).indices
        destination = (
            torch.arange(nodes, device=coordinates.device)
            .view(1, nodes, 1)
            .expand(batch, nodes, neighbours)
        )
        source = nearest

        customers = torch.arange(1, nodes, device=coordinates.device).view(1, -1).expand(batch, -1)
        depot = torch.zeros_like(customers)
        source = torch.cat((source.reshape(batch, -1), depot, customers), dim=1)
        destination = torch.cat((destination.reshape(batch, -1), customers, depot), dim=1)
        return source, destination

    @staticmethod
    def _gather_coordinates(coordinates: Tensor, index: Tensor) -> Tensor:
        return coordinates.gather(1, index.unsqueeze(-1).expand(-1, -1, 2))

    def raw_geometric_edge_features(
        self,
        coordinates: Float[Tensor, "b n 2"],
        source: Int[Tensor, "b e"],
        destination: Int[Tensor, "b e"],
    ) -> Float[Tensor, "b e f"]:
        """Build edge features in the documented order, with cos before sin."""

        source_xy = self._gather_coordinates(coordinates, source)
        destination_xy = self._gather_coordinates(coordinates, destination)
        delta = destination_xy - source_xy
        raw_distance = torch.linalg.vector_norm(delta, dim=-1)
        maximum = torch.cdist(coordinates, coordinates).amax(dim=(1, 2)).clamp_min(1e-6)
        normalized = raw_distance / maximum.unsqueeze(-1)
        distance_features = self.distance_encoding(normalized)
        angle = torch.atan2(delta[..., 1], delta[..., 0])
        geometry = torch.cat(
            (
                distance_features,
                torch.cos(angle).unsqueeze(-1),
                torch.sin(angle).unsqueeze(-1),
            ),
            dim=-1,
        )
        return geometry

    def geometric_edge_features(
        self,
        coordinates: Float[Tensor, "b n 2"],
        source: Int[Tensor, "b e"],
        destination: Int[Tensor, "b e"],
    ) -> Float[Tensor, "b e d"]:
        return self.edge_projection(
            self.raw_geometric_edge_features(coordinates, source, destination)
        )

    def forward(
        self, features: Float[Tensor, "b n d_in"]
    ) -> tuple[Float[Tensor, "b n d"], Float[Tensor, "b d"]]:
        if features.ndim != 3 or features.shape[-1] != self.in_dim:
            raise ValueError(
                f"expected features with shape (batch, nodes, {self.in_dim}), got {tuple(features.shape)}"
            )
        coordinates = features[..., :2]
        source, destination = self.sparse_edge_index(coordinates)
        edge_features = self.geometric_edge_features(coordinates, source, destination)
        initial = self.embed(features)
        hidden = initial
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, source, destination, edge_features)
            if index + 1 < len(self.blocks):
                hidden = F.relu(hidden)
                hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        if self.residual:
            hidden = hidden + initial
        return hidden, pool_mean(hidden)


__all__ = ["GNNEncoder"]
