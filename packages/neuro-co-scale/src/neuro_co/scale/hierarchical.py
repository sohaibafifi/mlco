"""Tensor utilities for hierarchical latent VRP decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from .types import Partition

ClusterSelectionMode = Literal["greedy", "sample"]


@dataclass(frozen=True, slots=True)
class ClusterIndex:
    """Static global-id cluster index used by hierarchical decoders."""

    clusters: tuple[tuple[int, ...], ...]
    num_nodes: int
    customer_to_cluster: tuple[int, ...]

    @property
    def num_clusters(self) -> int:
        return len(self.clusters)

    @property
    def max_cluster_size(self) -> int:
        return max((len(cluster) for cluster in self.clusters), default=0)

    @property
    def max_local_actions(self) -> int:
        return self.max_cluster_size + 1

    def cluster_of(self, customer: int) -> int:
        if customer <= 0 or customer >= self.num_nodes:
            raise ValueError(f"customer id {customer} is outside 1..{self.num_nodes - 1}")
        cluster = self.customer_to_cluster[customer]
        if cluster < 0:
            raise ValueError(f"customer id {customer} is not assigned to any cluster")
        return cluster


@dataclass(frozen=True, slots=True)
class SelectedClusterActionSet:
    """Padded selected-cluster action tensors for a batch."""

    node_embs: Tensor
    global_indices: Tensor
    mask: Tensor


@runtime_checkable
class ClusterSelector(Protocol):
    """Policy contract for selecting an active cluster from cluster embeddings."""

    def __call__(self, cluster_embs: Tensor, active_mask: Tensor) -> Tensor:
        """Return masked logits with shape `(batch, num_clusters)`."""
        ...


class LinearClusterSelector(nn.Module):
    """Independent linear scoring head over cluster embeddings."""

    def __init__(self, hidden_dim: int, *, bias: bool = True) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.score = nn.Linear(hidden_dim, 1, bias=bias)

    def forward(self, cluster_embs: Tensor, active_mask: Tensor) -> Tensor:
        _check_cluster_embeddings(cluster_embs)
        _check_cluster_mask(active_mask, cluster_embs)
        logits = self.score(cluster_embs).squeeze(-1)
        return mask_cluster_logits(logits, active_mask)


def build_cluster_index(partition: Partition, *, num_nodes: int) -> ClusterIndex:
    """Build a static customer-to-cluster index from global customer ids."""

    if num_nodes < 2:
        raise ValueError("num_nodes must include depot and at least one customer")
    customer_to_cluster = [-1] * num_nodes
    for cluster_idx, cluster in enumerate(partition.clusters):
        if not cluster:
            raise ValueError("clusters must not be empty")
        for customer in cluster:
            if customer == 0:
                raise ValueError("clusters must not include depot id 0")
            if customer < 0 or customer >= num_nodes:
                raise ValueError(f"customer id {customer} is outside the instance")
            if customer_to_cluster[customer] != -1:
                raise ValueError(f"duplicate customer id {customer}")
            customer_to_cluster[customer] = cluster_idx
    missing = [customer for customer in range(1, num_nodes) if customer_to_cluster[customer] < 0]
    if missing:
        raise ValueError(f"partition is missing {len(missing)} customers")
    return ClusterIndex(
        clusters=partition.clusters,
        num_nodes=num_nodes,
        customer_to_cluster=tuple(customer_to_cluster),
    )


def aggregate_cluster_embeddings(node_embs: Tensor, cluster_index: ClusterIndex) -> Tensor:
    """Mean-pool customer embeddings into cluster embeddings.

    The depot embedding is intentionally excluded. The returned tensor has
    shape `(batch, num_clusters, hidden_dim)`.
    """

    _check_node_embeddings(node_embs, cluster_index)
    if cluster_index.num_clusters == 0:
        raise ValueError("cluster_index must contain at least one cluster")
    pooled = [
        node_embs[:, torch.as_tensor(cluster, dtype=torch.long, device=node_embs.device)].mean(
            dim=1
        )
        for cluster in cluster_index.clusters
    ]
    return torch.stack(pooled, dim=1)


def active_cluster_mask(customer_mask: Tensor, cluster_index: ClusterIndex) -> Tensor:
    """Return which clusters contain at least one active customer.

    `customer_mask` is a global node mask with shape `(batch, num_nodes)`.
    Depot entries are ignored.
    """

    _check_global_mask(customer_mask, cluster_index)
    active = [
        customer_mask[
            :, torch.as_tensor(cluster, dtype=torch.long, device=customer_mask.device)
        ].any(dim=1)
        for cluster in cluster_index.clusters
    ]
    return torch.stack(active, dim=1)


def mask_cluster_logits(
    logits: Tensor,
    active_mask: Tensor,
    *,
    fill_value: float = -1.0e9,
) -> Tensor:
    """Mask inactive cluster logits with a large negative value."""

    if logits.ndim != 2:
        raise ValueError("cluster logits must have shape [batch, num_clusters]")
    if active_mask.shape != logits.shape:
        raise ValueError("active_mask shape must match cluster logits")
    return torch.where(active_mask, logits, logits.new_full((), fill_value))


def select_cluster(
    logits: Tensor,
    active_mask: Tensor,
    *,
    mode: ClusterSelectionMode = "greedy",
    generator: torch.Generator | None = None,
    validate: bool = True,
) -> Tensor:
    """Select a cluster from masked cluster logits.

    ``validate`` runs an active-cluster check that reads a scalar back to the
    host (a device sync). The hot decode loop already guarantees an active
    cluster per row, so it passes ``validate=False`` to keep the step sync-free.
    """

    if validate:
        _check_has_active_cluster(active_mask)
    masked_logits = mask_cluster_logits(logits, active_mask)
    if mode == "greedy":
        return masked_logits.argmax(dim=-1)
    if mode == "sample":
        probs = masked_logits.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    raise ValueError(f"unknown cluster selection mode {mode!r}")


def cluster_log_prob(logits: Tensor, selected_cluster: Tensor, active_mask: Tensor) -> Tensor:
    """Log-probability of selected clusters under masked cluster logits."""

    if selected_cluster.ndim != 1 or selected_cluster.shape[0] != logits.shape[0]:
        raise ValueError("selected_cluster must have shape [batch]")
    masked_logits = mask_cluster_logits(logits, active_mask)
    return masked_logits.log_softmax(dim=-1).gather(1, selected_cluster.unsqueeze(1)).squeeze(1)


def gather_selected_cluster_action_set(
    node_embs: Tensor,
    cluster_index: ClusterIndex,
    selected_cluster: Tensor,
    *,
    global_action_mask: Tensor | None = None,
) -> SelectedClusterActionSet:
    """Gather `depot + selected cluster customers` into a padded local action set."""

    _check_node_embeddings(node_embs, cluster_index)
    if selected_cluster.ndim != 1 or selected_cluster.shape[0] != node_embs.shape[0]:
        raise ValueError("selected_cluster must have shape [batch]")
    if global_action_mask is not None:
        _check_global_mask(global_action_mask, cluster_index)
        if global_action_mask.shape[0] != node_embs.shape[0]:
            raise ValueError("global_action_mask batch size must match node_embs")

    batch = node_embs.shape[0]
    max_actions = cluster_index.max_local_actions
    if max_actions <= 1:
        raise ValueError("cluster_index must contain at least one customer")

    indices = torch.zeros(batch, max_actions, dtype=torch.long, device=node_embs.device)
    mask = torch.zeros(batch, max_actions, dtype=torch.bool, device=node_embs.device)
    mask[:, 0] = True
    if global_action_mask is not None:
        mask[:, 0] = global_action_mask[:, 0]

    selected_list = selected_cluster.detach().cpu().tolist()
    for batch_idx, cluster_idx_raw in enumerate(selected_list):
        cluster_idx = int(cluster_idx_raw)
        if cluster_idx < 0 or cluster_idx >= cluster_index.num_clusters:
            raise ValueError(f"selected cluster {cluster_idx} is outside the cluster index")
        customers = cluster_index.clusters[cluster_idx]
        end = 1 + len(customers)
        indices[batch_idx, 1:end] = torch.as_tensor(
            customers, dtype=torch.long, device=node_embs.device
        )
        mask[batch_idx, 1:end] = True
        if global_action_mask is not None:
            mask[batch_idx, 1:end] &= global_action_mask[batch_idx, indices[batch_idx, 1:end]]

    gathered = node_embs.gather(
        1,
        indices.unsqueeze(-1).expand(batch, max_actions, node_embs.shape[-1]),
    )
    return SelectedClusterActionSet(node_embs=gathered, global_indices=indices, mask=mask)


def gather_neighborhood_action_set(
    node_embs: Tensor,
    cluster_index: ClusterIndex,
    selected_cluster: Tensor,
    *,
    neighbor_span: int,
    global_action_mask: Tensor | None = None,
) -> SelectedClusterActionSet:
    """Gather ``depot + anchor cluster + Morton-neighbor clusters`` into a slice.

    The action set for a route anchored at cluster ``c`` covers ``c`` plus the
    clusters ``c - neighbor_span .. c + neighbor_span`` in cluster (Morton) order.
    This lets a route cross cluster boundaries while keeping the action set
    bounded by ``(2 * neighbor_span + 1) * max_cluster_size + 1`` instead of the
    instance size.
    """

    _check_node_embeddings(node_embs, cluster_index)
    if neighbor_span <= 0:
        raise ValueError("neighbor_span must be positive")
    if selected_cluster.ndim != 1 or selected_cluster.shape[0] != node_embs.shape[0]:
        raise ValueError("selected_cluster must have shape [batch]")
    if global_action_mask is not None:
        _check_global_mask(global_action_mask, cluster_index)
        if global_action_mask.shape[0] != node_embs.shape[0]:
            raise ValueError("global_action_mask batch size must match node_embs")

    batch = node_embs.shape[0]
    num_clusters = cluster_index.num_clusters
    span_clusters = 2 * neighbor_span + 1
    max_actions = span_clusters * cluster_index.max_cluster_size + 1
    if max_actions <= 1:
        raise ValueError("cluster_index must contain at least one customer")

    indices = torch.zeros(batch, max_actions, dtype=torch.long, device=node_embs.device)
    mask = torch.zeros(batch, max_actions, dtype=torch.bool, device=node_embs.device)
    mask[:, 0] = True
    if global_action_mask is not None:
        mask[:, 0] = global_action_mask[:, 0]

    selected_list = selected_cluster.detach().cpu().tolist()
    for batch_idx, cluster_idx_raw in enumerate(selected_list):
        cluster_idx = int(cluster_idx_raw)
        if cluster_idx < 0 or cluster_idx >= num_clusters:
            raise ValueError(f"selected cluster {cluster_idx} is outside the cluster index")
        start = max(0, cluster_idx - neighbor_span)
        stop = min(num_clusters, cluster_idx + neighbor_span + 1)
        members: list[int] = []
        for neighbor in range(start, stop):
            members.extend(cluster_index.clusters[neighbor])
        end = 1 + len(members)
        member_tensor = torch.as_tensor(members, dtype=torch.long, device=node_embs.device)
        indices[batch_idx, 1:end] = member_tensor
        mask[batch_idx, 1:end] = True
        if global_action_mask is not None:
            mask[batch_idx, 1:end] &= global_action_mask[batch_idx, member_tensor]

    gathered = node_embs.gather(
        1,
        indices.unsqueeze(-1).expand(batch, max_actions, node_embs.shape[-1]),
    )
    return SelectedClusterActionSet(node_embs=gathered, global_indices=indices, mask=mask)


def scatter_local_logits(
    local_logits: Tensor,
    action_set: SelectedClusterActionSet,
    *,
    num_nodes: int,
    fill_value: float = -1.0e9,
) -> Tensor:
    """Scatter selected-cluster logits back to global node logits."""

    if local_logits.shape != action_set.mask.shape:
        raise ValueError("local_logits shape must match action_set mask")
    # Padded action slots default to global index 0 (the depot). Scattering them
    # straight into column 0 would overwrite the real depot logit whenever a
    # selected cluster is smaller than the padded width. Route every non-depot
    # slot that still maps to index 0 into a scratch column, then drop it.
    batch, max_actions = local_logits.shape
    positions = torch.arange(max_actions, device=local_logits.device).expand(batch, max_actions)
    is_pad = (action_set.global_indices == 0) & (positions != 0)
    scatter_indices = torch.where(
        is_pad, action_set.global_indices.new_full((), num_nodes), action_set.global_indices
    )
    out = local_logits.new_full((batch, num_nodes + 1), fill_value)
    safe_logits = torch.where(action_set.mask, local_logits, local_logits.new_full((), fill_value))
    out.scatter_(1, scatter_indices, safe_logits)
    return out[:, :num_nodes]


def _check_node_embeddings(node_embs: Tensor, cluster_index: ClusterIndex) -> None:
    if node_embs.ndim != 3:
        raise ValueError("node_embs must have shape [batch, num_nodes, hidden_dim]")
    if node_embs.shape[1] != cluster_index.num_nodes:
        raise ValueError(
            f"node_embs has {node_embs.shape[1]} nodes, expected {cluster_index.num_nodes}"
        )


def _check_global_mask(mask: Tensor, cluster_index: ClusterIndex) -> None:
    if mask.ndim != 2:
        raise ValueError("global mask must have shape [batch, num_nodes]")
    if mask.shape[1] != cluster_index.num_nodes:
        raise ValueError(
            f"global mask has {mask.shape[1]} nodes, expected {cluster_index.num_nodes}"
        )


def _check_cluster_embeddings(cluster_embs: Tensor) -> None:
    if cluster_embs.ndim != 3:
        raise ValueError("cluster_embs must have shape [batch, num_clusters, hidden_dim]")


def _check_cluster_mask(mask: Tensor, cluster_embs: Tensor) -> None:
    if mask.ndim != 2:
        raise ValueError("cluster mask must have shape [batch, num_clusters]")
    if mask.shape != cluster_embs.shape[:2]:
        raise ValueError("cluster mask shape must match cluster embeddings")


def _check_has_active_cluster(active_mask: Tensor) -> None:
    if active_mask.ndim != 2:
        raise ValueError("active_mask must have shape [batch, num_clusters]")
    if not bool(active_mask.any(dim=1).all().item()):
        raise ValueError("each batch item must have at least one active cluster")
