"""Local and cluster-level attention for large VRP instances.

Customers attend within their partition, then a learned pool produces one
token per cluster. Cluster tokens attend to each other and broadcast their
updates back to the customers. With at most ``m`` customers in each of ``C``
clusters, local attention costs ``O(n * m)`` and cluster attention costs
``O(C^2)``. The decoder consumes the cluster embeddings directly.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .hierarchical import ClusterIndex

_NEG_INF = -1.0e9


class _MaskedMHABlock(nn.Module):
    """Pre-norm multi-head self-attention + FFN with optional key padding."""

    def __init__(self, d: int, heads: int, ff_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if d % heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(
            nn.Linear(d, ff_mult * d),
            nn.GELU(),
            nn.Linear(ff_mult * d, d),
        )
        self.heads = heads
        self.dropout = dropout

    def forward(self, x: Tensor, key_padding: Tensor | None = None) -> Tensor:
        h = self.norm1(x)
        b, s, d = h.shape
        qkv = self.qkv(h).reshape(b, s, 3, self.heads, d // self.heads)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        attn_mask = None
        if key_padding is not None:
            # additive mask over keys, broadcast across heads and queries
            attn_mask = torch.zeros(b, 1, 1, s, dtype=h.dtype, device=h.device)
            attn_mask = attn_mask.masked_fill(key_padding[:, None, None, :], _NEG_INF)
        att = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        att = att.transpose(1, 2).reshape(b, s, d)
        x = x + self.proj(att)
        x = x + self.ff(self.norm2(x))
        return x


class _ClusterLocalLayer(nn.Module):
    """One round of intra-cluster, pool, inter-cluster, and broadcast updates."""

    def __init__(self, d: int, heads: int, ff_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.intra = _MaskedMHABlock(d, heads, ff_mult, dropout)
        self.inter = _MaskedMHABlock(d, heads, ff_mult, dropout)
        self.pool_query = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.pool_query, std=d**-0.5)
        self.broadcast_norm = nn.LayerNorm(d)

    def forward(
        self,
        h: Tensor,
        member_index: Tensor,
        member_mask: Tensor,
        customer_cluster: Tensor,
    ) -> tuple[Tensor, Tensor]:
        b, _, d = h.shape
        num_clusters, max_members = member_index.shape

        # 1. intra-cluster attention over padded cluster member sets
        members = h[:, member_index]  # (b, C, m, d)
        key_padding = (~member_mask).reshape(1, num_clusters, max_members).expand(b, -1, -1)
        updated = self.intra(
            members.reshape(b * num_clusters, max_members, d),
            key_padding.reshape(b * num_clusters, max_members),
        ).reshape(b, num_clusters, max_members, d)
        valid_ids = member_index[member_mask]  # (K,) every customer exactly once
        valid_updates = updated[:, member_mask]  # (b, K, d)
        h = h.index_copy(1, valid_ids, valid_updates)

        # 2. attention-pool aggregation into cluster embeddings
        members = h[:, member_index]
        scores = (members @ self.pool_query) / math.sqrt(d)  # (b, C, m)
        scores = scores.masked_fill(~member_mask.unsqueeze(0), _NEG_INF)
        weights = scores.softmax(dim=-1).unsqueeze(-1)
        cluster_emb = (weights * members).sum(dim=2)  # (b, C, d)

        # 3. inter-cluster attention with a depot token
        tokens = torch.cat([h[:, :1], cluster_emb], dim=1)  # (b, C+1, d)
        tokens = self.inter(tokens)
        depot_token = tokens[:, :1]
        cluster_emb = tokens[:, 1:]

        # 4. broadcast cluster context back to its customers (residual). Build the
        # tensor with a concat rather than in-place slice writes so autograd can
        # backprop through the rollout.
        customers = h[:, 1:] + cluster_emb[:, customer_cluster[1:]]
        depot = h[:, :1] + depot_token
        out = torch.cat([depot, customers], dim=1)
        return self.broadcast_norm(out), cluster_emb

    def forward_batched(
        self,
        h: Tensor,
        member_index: Tensor,  # (B, C, m) global ids, padded with 0
        member_mask: Tensor,  # (B, C, m) bool
        customer_cluster: Tensor,  # (B, N) cluster id per node, depot = -1
    ) -> tuple[Tensor, Tensor]:
        """Per-instance batched variant: each row has its own cluster layout."""

        batch, num_nodes, d = h.shape
        _, num_clusters, max_members = member_index.shape
        bidx = torch.arange(batch, device=h.device)[:, None, None]  # (B, 1, 1)

        # 1. intra-cluster attention over each instance's padded clusters
        members = h[bidx, member_index]  # (B, C, m, d)
        key_padding = (~member_mask).reshape(batch * num_clusters, max_members)
        updated = self.intra(
            members.reshape(batch * num_clusters, max_members, d), key_padding
        ).reshape(batch, num_clusters, max_members, d)
        # scatter members back; padded slots go to a scratch column (then dropped),
        # so the depot (never a member) is never overwritten.
        scratch = torch.full_like(member_index, num_nodes)
        flat_idx = torch.where(member_mask, member_index, scratch).reshape(
            batch, num_clusters * max_members
        )
        flat_upd = updated.reshape(batch, num_clusters * max_members, d)
        h_ext = torch.cat([h, h.new_zeros(batch, 1, d)], dim=1)
        h_ext = h_ext.scatter(1, flat_idx.unsqueeze(-1).expand(-1, -1, d), flat_upd)
        h = h_ext[:, :num_nodes]

        # 2. attention-pool aggregation (empty clusters -> 0 via nan_to_num)
        members = h[bidx, member_index]
        scores = (members @ self.pool_query) / math.sqrt(d)
        scores = scores.masked_fill(~member_mask, _NEG_INF)
        weights = torch.nan_to_num(scores.softmax(dim=-1), nan=0.0).unsqueeze(-1)
        cluster_emb = (weights * members).sum(dim=2)  # (B, C, d)

        # 3. inter-cluster attention with a depot token; mask empty clusters
        tokens = torch.cat([h[:, :1], cluster_emb], dim=1)  # (B, C+1, d)
        cluster_has_members = member_mask.any(dim=2)  # (B, C)
        inter_pad = torch.cat(
            [torch.zeros(batch, 1, dtype=torch.bool, device=h.device), ~cluster_has_members], dim=1
        )
        tokens = self.inter(tokens, inter_pad)
        depot_token = tokens[:, :1]
        cluster_emb = tokens[:, 1:]

        # 4. broadcast cluster context back to its customers (residual)
        cust_clusters = customer_cluster[:, 1:].clamp(min=0)  # (B, N-1)
        gathered = cluster_emb[torch.arange(batch, device=h.device)[:, None], cust_clusters]
        customers = h[:, 1:] + gathered
        depot = h[:, :1] + depot_token
        out = torch.cat([depot, customers], dim=1)
        return self.broadcast_norm(out), cluster_emb


class ClusterLocalEncoder(nn.Module):
    """Two-level cluster-local message-passing encoder.

    Implements the :class:`~neuro_co.scale.decode.ClusterAwareEncoder` contract:
    ``encode_clusters(features, cluster_index)`` returns
    ``(node_embs, cluster_embs, graph_emb)``.
    """

    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [_ClusterLocalLayer(hidden_dim, num_heads, ff_mult, dropout) for _ in range(num_layers)]
        )

    def encode_clusters(
        self, features: Tensor, cluster_index: ClusterIndex
    ) -> tuple[Tensor, Tensor, Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, num_nodes, in_dim]")
        if features.shape[1] != cluster_index.num_nodes:
            raise ValueError("features num_nodes must match cluster_index")
        member_index, member_mask, customer_cluster = _cluster_tensors(
            cluster_index, device=features.device
        )
        h = self.embed(features)
        cluster_emb = h.new_zeros(h.shape[0], cluster_index.num_clusters, h.shape[-1])
        for layer in self.layers:
            h, cluster_emb = layer(h, member_index, member_mask, customer_cluster)
        graph_emb = h.mean(dim=1)
        return h, cluster_emb, graph_emb

    def forward(
        self, features: Tensor, cluster_index: ClusterIndex
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.encode_clusters(features, cluster_index)

    def encode_clusters_batched(
        self, features: Tensor, cluster_indices: list[ClusterIndex]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode a batch of instances (one row each) with per-instance clusters.

        ``features`` has shape ``(B, N, in_dim)``; all instances share ``N`` and
        their clusters are padded to a common count and width. Returns
        ``(node_embs (B, N, d), cluster_embs (B, C_max, d), graph_emb (B, d))``.
        """

        if features.ndim != 3:
            raise ValueError("features must have shape [batch, num_nodes, in_dim]")
        if features.shape[0] != len(cluster_indices):
            raise ValueError("features batch must match the number of cluster indices")
        num_nodes = features.shape[1]
        if any(ci.num_nodes != num_nodes for ci in cluster_indices):
            raise ValueError("batched encode requires equal instance sizes")
        member_index, member_mask, customer_cluster = _cluster_tensors_batched(
            cluster_indices, device=features.device
        )
        h = self.embed(features)
        cluster_emb = h.new_zeros(h.shape[0], member_index.shape[1], h.shape[-1])
        for layer in self.layers:
            h, cluster_emb = layer.forward_batched(h, member_index, member_mask, customer_cluster)
        graph_emb = h.mean(dim=1)
        return h, cluster_emb, graph_emb


def _cluster_tensors(
    cluster_index: ClusterIndex, *, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    num_clusters = cluster_index.num_clusters
    max_members = cluster_index.max_cluster_size
    if num_clusters == 0 or max_members == 0:
        raise ValueError("cluster_index must contain at least one non-empty cluster")
    member_index = torch.zeros(num_clusters, max_members, dtype=torch.long, device=device)
    member_mask = torch.zeros(num_clusters, max_members, dtype=torch.bool, device=device)
    for cluster_idx, cluster in enumerate(cluster_index.clusters):
        size = len(cluster)
        member_index[cluster_idx, :size] = torch.as_tensor(cluster, dtype=torch.long, device=device)
        member_mask[cluster_idx, :size] = True
    customer_cluster = torch.as_tensor(
        cluster_index.customer_to_cluster, dtype=torch.long, device=device
    )
    return member_index, member_mask, customer_cluster


def _cluster_tensors_batched(
    cluster_indices: list[ClusterIndex], *, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    batch = len(cluster_indices)
    if batch == 0:
        raise ValueError("cluster_indices must not be empty")
    num_nodes = cluster_indices[0].num_nodes
    num_clusters = max(ci.num_clusters for ci in cluster_indices)
    max_members = max(ci.max_cluster_size for ci in cluster_indices)
    if num_clusters == 0 or max_members == 0:
        raise ValueError("cluster_indices must contain non-empty clusters")
    member_index = torch.zeros(batch, num_clusters, max_members, dtype=torch.long, device=device)
    member_mask = torch.zeros(batch, num_clusters, max_members, dtype=torch.bool, device=device)
    customer_cluster = torch.zeros(batch, num_nodes, dtype=torch.long, device=device)
    for b, cluster_index in enumerate(cluster_indices):
        customer_cluster[b] = torch.as_tensor(
            cluster_index.customer_to_cluster, dtype=torch.long, device=device
        )
        for cluster_idx, cluster in enumerate(cluster_index.clusters):
            size = len(cluster)
            member_index[b, cluster_idx, :size] = torch.as_tensor(
                cluster, dtype=torch.long, device=device
            )
            member_mask[b, cluster_idx, :size] = True
    return member_index, member_mask, customer_cluster
