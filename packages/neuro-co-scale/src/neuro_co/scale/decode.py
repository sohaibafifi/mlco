"""End-to-end hierarchical CVRP decode-step skeleton.

This is the first runnable version of the hierarchical latent decoder described
in ``hierarchical_decoder_and_sparse_refinement.md``. It is intentionally a thin
layer over ``neuro-co-core``:

- the node encoder (e.g. ``neuro_co.core.models.AMEncoder``) produces customer
  embeddings once;
- a :class:`~neuro_co.scale.hierarchical.ClusterSelector` chooses an active
  cluster from aggregated cluster embeddings;
- a local pointer (e.g. ``neuro_co.core.models.PointerDecoder``) decodes a route
  only over ``depot + customers of the selected cluster``.

The decoder never solves clusters independently. It keeps a single global latent
state, selects a cluster at each route start, and points only inside the
selected cluster. The local action set is bounded by the cluster size ``m``
rather than the instance size ``n``, which is the scaling mechanism.

The batch dimension is interpreted as independent rollouts (samples / POMO-style
starts) over the *same* instance and the *same* partition. This matches the
shared-``ClusterIndex`` assumption of :mod:`neuro_co.scale.hierarchical`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, cast, runtime_checkable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from .hierarchical import (
    ClusterIndex,
    ClusterSelector,
    SelectedClusterActionSet,
    aggregate_cluster_embeddings,
    build_cluster_index,
    cluster_log_prob,
    gather_neighborhood_action_set,
    gather_selected_cluster_action_set,
    select_cluster,
)
from .metrics import solution_cost
from .types import Partition, Solution, VRPInstance

DecodeMode = Literal["greedy", "sample"]

_NEG_INF = -1.0e9


@runtime_checkable
class NodeEncoder(Protocol):
    """Maps node features to node embeddings and a graph embedding."""

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(node_embs, graph_emb)`` for features ``(batch, num_nodes, in_dim)``."""
        ...


@runtime_checkable
class ClusterAwareEncoder(Protocol):
    """Encoder that emits cluster embeddings directly from a partition.

    Detected structurally (presence of ``encode_clusters``), so a cluster-local
    encoder and a plain node encoder are dispatched by capability rather than by
    a flag.
    """

    def encode_clusters(
        self, features: Tensor, cluster_index: ClusterIndex
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(node_embs, cluster_embs, graph_emb)``."""
        ...


@runtime_checkable
class LocalPointer(Protocol):
    """Pointer policy over a local action set (``neuro_co.core`` PointerDecoder)."""

    def __call__(
        self,
        node_embs: Tensor,
        graph_emb: Tensor,
        first_idx: Tensor,
        current_idx: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Return logits with shape ``(batch, num_local_actions)``."""
        ...


@dataclass(frozen=True, slots=True)
class ActionSetTables:
    """Static per-cluster action layout, precomputed once per rollout.

    ``index[c]`` lists the global node ids reachable from a route anchored at
    cluster ``c`` (column 0 is the depot, remaining columns are members, padded
    with 0). ``valid[c]`` marks the real (non-padding) action slots.
    """

    index: Tensor  # (num_clusters, max_actions)
    valid: Tensor  # (num_clusters, max_actions)


@runtime_checkable
class ActionSetBuilder(Protocol):
    """Builds the local action set for a route anchored at a selected cluster."""

    def __call__(
        self,
        node_embs: Tensor,
        cluster_index: ClusterIndex,
        selected_cluster: Tensor,
        *,
        global_action_mask: Tensor | None = None,
    ) -> SelectedClusterActionSet: ...

    def tables(self, cluster_index: ClusterIndex, *, device: torch.device) -> ActionSetTables:
        """Precompute the static (index, valid) layout used by the fast rollout."""
        ...


@dataclass(frozen=True, slots=True)
class ClusterActionSet:
    """S1: confine a route to its selected cluster (depot + that cluster)."""

    def __call__(
        self,
        node_embs: Tensor,
        cluster_index: ClusterIndex,
        selected_cluster: Tensor,
        *,
        global_action_mask: Tensor | None = None,
    ) -> SelectedClusterActionSet:
        return gather_selected_cluster_action_set(
            node_embs, cluster_index, selected_cluster, global_action_mask=global_action_mask
        )

    def tables(self, cluster_index: ClusterIndex, *, device: torch.device) -> ActionSetTables:
        return _build_action_tables(
            cluster_index,
            max_actions=cluster_index.max_local_actions,
            members_of=lambda c: cluster_index.clusters[c],
            device=device,
        )


@dataclass(frozen=True, slots=True)
class NeighborhoodActionSet:
    """S2: let a route cross into Morton-neighbor clusters of the anchor."""

    neighbor_span: int = 1

    def __call__(
        self,
        node_embs: Tensor,
        cluster_index: ClusterIndex,
        selected_cluster: Tensor,
        *,
        global_action_mask: Tensor | None = None,
    ) -> SelectedClusterActionSet:
        return gather_neighborhood_action_set(
            node_embs,
            cluster_index,
            selected_cluster,
            neighbor_span=self.neighbor_span,
            global_action_mask=global_action_mask,
        )

    def tables(self, cluster_index: ClusterIndex, *, device: torch.device) -> ActionSetTables:
        k = self.neighbor_span
        num_clusters = cluster_index.num_clusters
        max_actions = (2 * k + 1) * cluster_index.max_cluster_size + 1

        def members_of(c: int) -> tuple[int, ...]:
            members: list[int] = []
            for neighbor in range(max(0, c - k), min(num_clusters, c + k + 1)):
                members.extend(cluster_index.clusters[neighbor])
            return tuple(members)

        return _build_action_tables(
            cluster_index, max_actions=max_actions, members_of=members_of, device=device
        )


def _build_action_tables(
    cluster_index: ClusterIndex,
    *,
    max_actions: int,
    members_of: Callable[[int], tuple[int, ...]],
    device: torch.device,
) -> ActionSetTables:
    num_clusters = cluster_index.num_clusters
    index = torch.zeros(num_clusters, max_actions, dtype=torch.long, device=device)
    valid = torch.zeros(num_clusters, max_actions, dtype=torch.bool, device=device)
    valid[:, 0] = True  # depot slot
    for cluster in range(num_clusters):
        members = members_of(cluster)
        end = 1 + len(members)
        if end > max_actions:
            raise ValueError("action set width exceeded; check neighbor span and cluster sizes")
        index[cluster, 1:end] = torch.as_tensor(members, dtype=torch.long, device=device)
        valid[cluster, 1:end] = True
    return ActionSetTables(index=index, valid=valid)


@dataclass(frozen=True, slots=True)
class HierarchicalRollout:
    """Result of a hierarchical decode rollout over one instance."""

    solutions: tuple[Solution, ...]
    costs: Tensor
    cluster_log_prob: Tensor
    node_log_prob: Tensor
    best_index: int

    @property
    def log_prob(self) -> Tensor:
        """Per-sample total log-probability of cluster and node decisions."""

        return self.cluster_log_prob + self.node_log_prob

    @property
    def best_solution(self) -> Solution:
        return self.solutions[self.best_index]

    @property
    def best_cost(self) -> float:
        return float(self.costs[self.best_index])


class HierarchicalPolicy(nn.Module):
    """Compose an encoder, a cluster selector, and a local pointer.

    The three components are injected, not selected by flags, so any
    ``neuro-co-core`` encoder/decoder or a custom one can be plugged in.
    """

    def __init__(
        self,
        encoder: NodeEncoder | ClusterAwareEncoder,
        cluster_selector: ClusterSelector,
        local_pointer: LocalPointer,
        action_set_builder: ActionSetBuilder | None = None,
        *,
        reanchor: bool = False,
        use_checkpoint: bool = False,
        checkpoint_chunk: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.cluster_selector = cluster_selector
        self.local_pointer = local_pointer
        # Default S1: a route stays inside its selected cluster.
        self.action_set_builder = action_set_builder or ClusterActionSet()
        # S3: re-anchor an open route to a new cluster when its neighborhood is
        # exhausted, instead of forcing it to close (the depot stays available,
        # so the policy still chooses between closing and continuing).
        self.reanchor = reanchor
        # Gradient checkpointing of the decode loop (memory for large-n training).
        self.use_checkpoint = use_checkpoint
        self.checkpoint_chunk = checkpoint_chunk

    @property
    def device(self) -> torch.device:
        param = next(self.parameters(), None)
        return torch.device("cpu") if param is None else param.device

    @torch.no_grad()
    def solve(
        self,
        instance: VRPInstance,
        partition: Partition,
        *,
        num_samples: int = 1,
        mode: DecodeMode = "greedy",
        generator: torch.Generator | None = None,
    ) -> Solution:
        """Convenience wrapper that returns only the best-cost solution."""

        return self.rollout(
            instance,
            partition,
            num_samples=num_samples,
            mode=mode,
            generator=generator,
        ).best_solution

    def rollout(
        self,
        instance: VRPInstance,
        partition: Partition,
        *,
        num_samples: int = 1,
        mode: DecodeMode = "greedy",
        generator: torch.Generator | None = None,
    ) -> HierarchicalRollout:
        """Run a full hierarchical decode over one instance and partition."""

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        cluster_index = build_cluster_index(partition, num_nodes=instance.num_nodes)
        return hierarchical_rollout(
            instance,
            cluster_index,
            encoder=self.encoder,
            cluster_selector=self.cluster_selector,
            local_pointer=self.local_pointer,
            action_set_builder=self.action_set_builder,
            reanchor=self.reanchor,
            use_checkpoint=self.use_checkpoint,
            checkpoint_chunk=self.checkpoint_chunk,
            num_samples=num_samples,
            mode=mode,
            generator=generator,
            device=self.device,
        )

    def rollout_batched(
        self,
        instances: list[VRPInstance],
        partitions: list[Partition],
        *,
        num_samples: int = 1,
        mode: DecodeMode = "greedy",
        generator: torch.Generator | None = None,
    ) -> BatchedRollout:
        """Decode a batch of instances at once (one batched decode loop)."""

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        cluster_indices = [
            build_cluster_index(partition, num_nodes=instance.num_nodes)
            for instance, partition in zip(instances, partitions, strict=True)
        ]
        return hierarchical_rollout_batched(
            instances,
            cluster_indices,
            encoder=self.encoder,
            cluster_selector=self.cluster_selector,
            local_pointer=self.local_pointer,
            action_set_builder=self.action_set_builder,
            reanchor=self.reanchor,
            use_checkpoint=self.use_checkpoint,
            checkpoint_chunk=self.checkpoint_chunk,
            num_samples=num_samples,
            mode=mode,
            generator=generator,
            device=self.device,
        )


def _encode_with_clusters(
    encoder: NodeEncoder | ClusterAwareEncoder,
    features: Tensor,
    cluster_index: ClusterIndex,
) -> tuple[Tensor, Tensor, Tensor]:
    """Encode nodes and clusters, using cluster-aware encoders when available."""

    if isinstance(encoder, ClusterAwareEncoder):
        return encoder.encode_clusters(features, cluster_index)
    node_embs, graph_emb = cast(NodeEncoder, encoder)(features)
    cluster_embs = aggregate_cluster_embeddings(node_embs, cluster_index)
    return node_embs, cluster_embs, graph_emb


def build_instance_features(instance: VRPInstance) -> Tensor:
    """Build core-style CVRP node features ``[x, y, demand / capacity]``.

    Coordinates are min-max normalized into the unit square (aspect ratio kept)
    so a policy trained on synthetic [0,1] instances transfers to CVRPLIB's
    [0,1000] integer grid. Only the model input is rescaled; costs stay in the
    instance's own units, which is what BKS comparisons require.
    """

    coords = torch.as_tensor(instance.coords, dtype=torch.float32)
    low = coords.min(dim=0).values
    span = (coords.max(dim=0).values - low).max().clamp_min(1e-9)
    coords = (coords - low) / span
    demand = torch.as_tensor(instance.demand, dtype=torch.float32) / float(instance.capacity)
    return torch.cat([coords, demand.unsqueeze(-1)], dim=-1)


def _decode_loop(
    *,
    node_embs: Tensor,
    base_cluster_logits: Tensor,
    graph_emb: Tensor,
    index_table: Tensor,
    valid_table: Tensor,
    demand: Tensor,
    customer_to_cluster: Tensor,
    cluster_sizes: Tensor,
    full_capacity: Tensor,
    num_nodes: int,
    local_pointer: LocalPointer,
    reanchor: bool,
    mode: DecodeMode,
    generator: torch.Generator | None,
    use_checkpoint: bool = False,
    checkpoint_chunk: int = 64,
) -> tuple[list[list[tuple[int, ...]]], Tensor, Tensor]:
    """Shared sync-free decode loop over R rows (samples and/or instances).

    Every input carries a leading row dimension R, so the single-instance path
    (R = samples, tables broadcast) and the batched path (R = instances*samples,
    per-row tables) run the exact same loop. With ``use_checkpoint`` the loop is
    run in gradient-checkpointed chunks (activations recomputed in backward), so
    memory is bounded by one chunk instead of the whole rollout. Checkpointed
    sampling preserves the global RNG or snapshots an explicitly supplied
    generator per chunk, so recomputation draws the same actions.
    """

    if checkpoint_chunk <= 0:
        raise ValueError("checkpoint_chunk must be positive")
    rows_total = node_embs.shape[0]
    device = node_embs.device
    hidden = node_embs.shape[-1]
    num_clusters = index_table.shape[1]
    rows = torch.arange(rows_total, device=device)
    col0_mask = torch.zeros(1, num_clusters, dtype=torch.bool, device=device)
    col0_mask[0, 0] = True
    false_col = torch.zeros(rows_total, 1, dtype=torch.bool, device=device)
    zero = node_embs.new_zeros((), dtype=torch.float32)
    # These masks/counts are private rollout state. Autograd and checkpoint
    # recomputation need their earlier values; inference does not.
    inplace_state = not torch.is_grad_enabled() and not use_checkpoint
    cpu_greedy = inplace_state and mode == "greedy" and device.type == "cpu"

    def step(
        t_node_embs: Tensor,
        t_base_logits: Tensor,
        t_graph_emb: Tensor,
        state: tuple[Tensor, ...],
        gen: torch.Generator | None,
    ) -> tuple[tuple[Tensor, ...], Tensor, Tensor]:
        (
            remaining,
            remaining_count,
            capacity_left,
            current,
            selected_cluster,
            route_open,
            route_first,
            has_customer,
            cluster_logp,
            node_logp,
        ) = state
        remaining_per_row = remaining_count.sum(dim=1)
        running = (remaining_per_row > 0) | route_open
        has_remaining = remaining_per_row > 0
        if reanchor:
            cur_idx = index_table[rows, selected_cluster]
            cur_feasible = (
                valid_table[rows, selected_cluster]
                & remaining.gather(1, cur_idx)
                & (demand.gather(1, cur_idx) <= capacity_left.unsqueeze(1) + 1e-6)
            )[:, 1:].any(dim=1)
            need_new = ((~route_open) | (route_open & ~cur_feasible)) & has_remaining
        else:
            need_new = (~route_open) & has_remaining
        fresh_route = need_new & ~route_open

        # On CPU greedy inference, avoid a full cluster normalization when no
        # row needs a new anchor. Do not add a device synchronization on GPU,
        # or change the random-number stream of sampled rollouts.
        if not cpu_greedy or bool(need_new.any()):
            active = remaining_count > 0
            no_active = ~active.any(dim=1, keepdim=True)
            sane_active = active | (no_active & col0_mask)
            chosen = select_cluster(
                t_base_logits, sane_active, mode=mode, generator=gen, validate=False
            )
            chosen_logp = cluster_log_prob(t_base_logits, chosen, sane_active)
            selected_cluster = torch.where(need_new, chosen, selected_cluster)
            cluster_logp = cluster_logp + torch.where(need_new, chosen_logp, zero)
        capacity_left = torch.where(fresh_route, full_capacity, capacity_left)
        route_open = route_open | need_new

        idx = index_table[rows, selected_cluster]
        val = valid_table[rows, selected_cluster]
        if reanchor:
            # Route endpoints are global node IDs. An S3 window can exclude
            # either endpoint after a reanchor, so expose their embeddings in
            # context-only slots while keeping them out of the action set.
            # This retains the existing pointer API and parameter shapes.
            first_local = torch.full_like(route_first, idx.shape[1])
            current_local = first_local + 1
            idx = torch.cat([idx, route_first.unsqueeze(1), current.unsqueeze(1)], dim=1)
            val = torch.cat([val, false_col, false_col], dim=1)
        else:
            first_local = (idx == route_first.unsqueeze(1)).float().argmax(dim=1)
            current_local = (idx == current.unsqueeze(1)).float().argmax(dim=1)
        width = idx.shape[1]
        node_local = t_node_embs.gather(1, idx.unsqueeze(-1).expand(rows_total, width, hidden))
        remaining_local = remaining.gather(1, idx)
        fits_local = demand.gather(1, idx) <= capacity_left.unsqueeze(1) + 1e-6
        mask = val & remaining_local & fits_local
        mask = torch.cat([has_customer.unsqueeze(1), mask[:, 1:]], dim=1)

        local_logits = local_pointer(node_local, t_graph_emb, first_local, current_local, mask)
        action_local = _select_action(local_logits, mask, mode=mode, generator=gen)
        step_logp = _action_log_prob(local_logits, mask, action_local)
        chosen_global = idx.gather(1, action_local.unsqueeze(1)).squeeze(1)
        node_logp = node_logp + torch.where(running, step_logp, zero)

        is_depot = chosen_global == 0
        visit = running & ~is_depot
        close = running & is_depot
        clear_target = torch.where(visit, chosen_global, torch.zeros_like(chosen_global))
        if inplace_state:
            remaining.scatter_(1, clear_target.unsqueeze(1), false_col)
        else:
            remaining = remaining.scatter(1, clear_target.unsqueeze(1), false_col)
        safe_cluster = (
            customer_to_cluster.gather(1, clear_target.unsqueeze(1)).squeeze(1).clamp(min=0)
        )
        if inplace_state:
            remaining_count.scatter_add_(1, safe_cluster.unsqueeze(1), (-visit.long()).unsqueeze(1))
        else:
            remaining_count = remaining_count.scatter_add(
                1, safe_cluster.unsqueeze(1), (-visit.long()).unsqueeze(1)
            )
        chosen_demand = demand.gather(1, chosen_global.unsqueeze(1)).squeeze(1)
        capacity_left = torch.where(visit, capacity_left - chosen_demand, capacity_left)
        capacity_left = torch.where(close, full_capacity, capacity_left)
        current = torch.where(visit, chosen_global, current)
        current = torch.where(close, torch.zeros_like(current), current)
        first_now = visit & ~has_customer
        route_first = torch.where(first_now, chosen_global, route_first)
        route_first = torch.where(close, torch.zeros_like(route_first), route_first)
        has_customer = (has_customer | visit) & ~close
        route_open = route_open & ~close

        new_state = (
            remaining,
            remaining_count,
            capacity_left,
            current,
            selected_cluster,
            route_open,
            route_first,
            has_customer,
            cluster_logp,
            node_logp,
        )
        return new_state, chosen_global, running

    state: tuple[Tensor, ...] = (
        torch.cat(
            [
                torch.zeros(rows_total, 1, dtype=torch.bool, device=device),
                torch.ones(rows_total, num_nodes - 1, dtype=torch.bool, device=device),
            ],
            dim=1,
        ),
        cluster_sizes.clone(),
        full_capacity.clone(),
        torch.zeros(rows_total, dtype=torch.long, device=device),
        torch.zeros(rows_total, dtype=torch.long, device=device),
        torch.zeros(rows_total, dtype=torch.bool, device=device),
        torch.zeros(rows_total, dtype=torch.long, device=device),
        torch.zeros(rows_total, dtype=torch.bool, device=device),
        torch.zeros(rows_total, dtype=torch.float32, device=device),
        torch.zeros(rows_total, dtype=torch.float32, device=device),
    )

    max_steps = 2 * num_nodes + num_clusters + 2
    action_seq = torch.zeros(rows_total, max_steps, dtype=torch.long, device=device)
    running_seq = torch.zeros(rows_total, max_steps, dtype=torch.bool, device=device)
    term_check = 256
    executed = max_steps

    def done(s: tuple[Tensor, ...]) -> bool:
        return not bool(((s[1].sum(dim=1) > 0) | s[5]).any())

    if use_checkpoint:

        def chunk_fn(
            n_steps: int,
            ne: Tensor,
            bl: Tensor,
            ge: Tensor,
            rng_state: Tensor | None,
            *flat_state: Tensor,
        ) -> tuple[Tensor, ...]:
            # Recreate the caller's stream from this chunk's immutable snapshot
            # on both forward and recomputation. Never rewind or advance the
            # caller's generator during backward.
            chunk_gen = None
            if rng_state is not None:
                assert generator is not None
                chunk_gen = torch.Generator(device=generator.device)
                chunk_gen.set_state(rng_state)
            chunk_state = flat_state
            acts, runs = [], []
            for _ in range(n_steps):
                chunk_state, act, run = step(ne, bl, ge, chunk_state, chunk_gen)
                acts.append(act)
                runs.append(run)
            result = (*chunk_state, torch.stack(acts, dim=1), torch.stack(runs, dim=1))
            return result if chunk_gen is None else (*result, chunk_gen.get_state())

        pos = 0
        while pos < max_steps:
            span = min(checkpoint_chunk, max_steps - pos)
            out = _torch_checkpoint(
                chunk_fn,
                span,
                node_embs,
                base_cluster_logits,
                graph_emb,
                None if generator is None else generator.get_state(),
                *state,
                use_reentrant=False,
                preserve_rng_state=True,
            )
            state = tuple(out[:10])
            action_seq[:, pos : pos + span] = out[10]
            running_seq[:, pos : pos + span] = out[11]
            if generator is not None:
                generator.set_state(out[12])
            pos += span
            if done(state):
                executed = pos
                break
    else:
        for t in range(max_steps):
            if t % term_check == 0 and done(state):
                executed = t
                break
            state, act, run = step(node_embs, base_cluster_logits, graph_emb, state, generator)
            action_seq[:, t] = act
            running_seq[:, t] = run

    if executed == max_steps and not done(state):
        raise RuntimeError("hierarchical decode exceeded the step budget; check feasibility")

    action_host = action_seq[:, :executed].cpu().tolist()
    running_host = running_seq[:, :executed].cpu().tolist()
    routes: list[list[tuple[int, ...]]] = [[] for _ in range(rows_total)]
    for b in range(rows_total):
        open_r: list[int] = []
        actions_b = action_host[b]
        runs_b = running_host[b]
        for tt in range(executed):
            if not runs_b[tt]:
                continue
            node = actions_b[tt]
            if node == 0:
                if open_r:
                    routes[b].append(tuple(open_r))
                    open_r = []
            else:
                open_r.append(node)
        if open_r:
            routes[b].append(tuple(open_r))
    return routes, state[8], state[9]


def hierarchical_rollout(
    instance: VRPInstance,
    cluster_index: ClusterIndex,
    *,
    encoder: NodeEncoder | ClusterAwareEncoder,
    cluster_selector: ClusterSelector,
    local_pointer: LocalPointer,
    action_set_builder: ActionSetBuilder | None = None,
    reanchor: bool = False,
    use_checkpoint: bool = False,
    checkpoint_chunk: int = 64,
    num_samples: int = 1,
    mode: DecodeMode = "greedy",
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> HierarchicalRollout:
    """Decode one instance with the hierarchical depot -> cluster -> route loop.

    Heavy tensor work (encode, cluster scoring, local pointer, slicing) is
    batched across the ``num_samples`` rollouts. Per-rollout route bookkeeping is
    a small Python loop, which keeps the skeleton readable; ``num_samples`` is
    expected to stay small.
    """

    if cluster_index.num_nodes != instance.num_nodes:
        raise ValueError("cluster_index num_nodes must match the instance")
    if action_set_builder is None:
        action_set_builder = ClusterActionSet()
    device = torch.device("cpu") if device is None else device
    demand_np = np.asarray(instance.demand, dtype=np.float64)
    if float(demand_np[1:].max(initial=0.0)) > instance.capacity + 1e-9:
        raise ValueError("instance has a customer demand greater than vehicle capacity")

    num_nodes = instance.num_nodes
    batch = num_samples

    features = build_instance_features(instance).to(device)
    features = features.unsqueeze(0).expand(batch, num_nodes, features.shape[-1])
    node_embs, cluster_embs, graph_emb = _encode_with_clusters(encoder, features, cluster_index)

    num_clusters = cluster_index.num_clusters
    layout = action_set_builder.tables(cluster_index, device=device)
    # Single instance: every row (sample) shares one partition, so broadcast the
    # static tables to a per-row shape and reuse the shared decode loop.
    index_table = layout.index.unsqueeze(0).expand(batch, num_clusters, layout.index.shape[1])
    valid_table = layout.valid.unsqueeze(0).expand(batch, num_clusters, layout.valid.shape[1])
    demand = (
        torch.as_tensor(demand_np, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .expand(batch, num_nodes)
    )
    customer_to_cluster = (
        torch.as_tensor(cluster_index.customer_to_cluster, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(batch, num_nodes)
    )
    cluster_sizes = (
        torch.tensor(
            [len(cluster) for cluster in cluster_index.clusters], dtype=torch.long, device=device
        )
        .unsqueeze(0)
        .expand(batch, num_clusters)
    )
    full_capacity = torch.full(
        (batch,), float(instance.capacity), dtype=torch.float32, device=device
    )
    base_cluster_logits = cluster_selector(
        cluster_embs, torch.ones(batch, num_clusters, dtype=torch.bool, device=device)
    )

    routes, cluster_logp, node_logp = _decode_loop(
        node_embs=node_embs,
        base_cluster_logits=base_cluster_logits,
        graph_emb=graph_emb,
        index_table=index_table,
        valid_table=valid_table,
        demand=demand,
        customer_to_cluster=customer_to_cluster,
        cluster_sizes=cluster_sizes,
        full_capacity=full_capacity,
        num_nodes=num_nodes,
        local_pointer=local_pointer,
        reanchor=reanchor,
        mode=mode,
        generator=generator,
        use_checkpoint=use_checkpoint,
        checkpoint_chunk=checkpoint_chunk,
    )

    solutions = tuple(
        Solution(
            routes=tuple(route for route in sample_routes if route),
            metadata={"source": "hierarchical_decode", "mode": mode, "sample": b},
        )
        for b, sample_routes in enumerate(routes)
    )
    costs = torch.tensor(
        [solution_cost(instance, solution) for solution in solutions], dtype=torch.float32
    )
    best_index = int(torch.argmin(costs))
    return HierarchicalRollout(
        solutions=solutions,
        costs=costs,
        cluster_log_prob=cluster_logp,
        node_log_prob=node_logp,
        best_index=best_index,
    )


@dataclass(frozen=True, slots=True)
class BatchedRollout:
    """Result of a batched rollout over ``B`` instances, ``S`` samples each."""

    solutions: tuple[tuple[Solution, ...], ...]  # (B,) tuples of S solutions
    costs: Tensor  # (B, S)
    cluster_log_prob: Tensor  # (B, S)
    node_log_prob: Tensor  # (B, S)

    @property
    def log_prob(self) -> Tensor:
        return self.cluster_log_prob + self.node_log_prob


def hierarchical_rollout_batched(
    instances: list[VRPInstance],
    cluster_indices: list[ClusterIndex],
    *,
    encoder: NodeEncoder | ClusterAwareEncoder,
    cluster_selector: ClusterSelector,
    local_pointer: LocalPointer,
    action_set_builder: ActionSetBuilder | None = None,
    reanchor: bool = False,
    use_checkpoint: bool = False,
    checkpoint_chunk: int = 64,
    num_samples: int = 1,
    mode: DecodeMode = "greedy",
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> BatchedRollout:
    """Decode ``B`` instances (``num_samples`` rollouts each) in one batched loop.

    Instances must share the customer count ``num_nodes`` (the training generator
    produces a fixed size); their partitions may differ and are padded to a common
    cluster count and action width. Each instance is encoded separately (cheap,
    once), and the ``B * num_samples`` rows run through the shared decode loop.
    """

    batch = len(instances)
    if batch == 0:
        raise ValueError("instances must not be empty")
    if len(cluster_indices) != batch:
        raise ValueError("instances and cluster_indices must have the same length")
    if action_set_builder is None:
        action_set_builder = ClusterActionSet()
    device = torch.device("cpu") if device is None else device

    num_nodes = instances[0].num_nodes
    for instance, cluster_index in zip(instances, cluster_indices, strict=True):
        if instance.num_nodes != num_nodes:
            raise ValueError("batched rollout requires equal instance sizes")
        if cluster_index.num_nodes != instance.num_nodes:
            raise ValueError("cluster_index num_nodes must match its instance")
        demand_np = np.asarray(instance.demand, dtype=np.float64)
        if float(demand_np[1:].max(initial=0.0)) > instance.capacity + 1e-9:
            raise ValueError("instance has a customer demand greater than vehicle capacity")

    layouts = [action_set_builder.tables(ci, device=device) for ci in cluster_indices]
    num_clusters = max(ci.num_clusters for ci in cluster_indices)
    action_width = max(layout.index.shape[1] for layout in layouts)

    feats_list = []
    index_list, valid_list, demand_list, c2c_list, size_list, cap_list = [], [], [], [], [], []
    for instance, cluster_index, layout in zip(instances, cluster_indices, layouts, strict=True):
        clusters_here = cluster_index.num_clusters
        feats_list.append(build_instance_features(instance).to(device).unsqueeze(0))  # (1, N, 3)

        idx_pad = torch.zeros(num_clusters, action_width, dtype=torch.long, device=device)
        val_pad = torch.zeros(num_clusters, action_width, dtype=torch.bool, device=device)
        idx_pad[:clusters_here, : layout.index.shape[1]] = layout.index
        val_pad[:clusters_here, : layout.valid.shape[1]] = layout.valid
        index_list.append(idx_pad.unsqueeze(0))
        valid_list.append(val_pad.unsqueeze(0))

        demand_list.append(
            torch.as_tensor(
                np.asarray(instance.demand), dtype=torch.float32, device=device
            ).unsqueeze(0)
        )
        c2c_list.append(
            torch.as_tensor(
                cluster_index.customer_to_cluster, dtype=torch.long, device=device
            ).unsqueeze(0)
        )
        sizes = [len(c) for c in cluster_index.clusters] + [0] * (num_clusters - clusters_here)
        size_list.append(torch.tensor(sizes, dtype=torch.long, device=device).unsqueeze(0))
        cap_list.append(float(instance.capacity))

    features = torch.cat(feats_list, dim=0)  # (B, N, 3)
    encode_batched = getattr(encoder, "encode_clusters_batched", None)
    if encode_batched is not None:
        # One batched encode (clusters padded internally) avoids the per-instance
        # encoder loop that otherwise serializes the GPU.
        node_embs, cluster_embs, graph_emb = encode_batched(features, cluster_indices)
    else:
        node_parts, cluster_parts, graph_parts = [], [], []
        for i, cluster_index in enumerate(cluster_indices):
            node_emb, cluster_emb, graph = _encode_with_clusters(
                encoder, feats_list[i], cluster_index
            )
            node_parts.append(node_emb)
            graph_parts.append(graph)
            padded = cluster_emb.new_zeros(1, num_clusters, cluster_emb.shape[-1])
            padded[:, : cluster_index.num_clusters] = cluster_emb
            cluster_parts.append(padded)
        node_embs = torch.cat(node_parts, dim=0)
        cluster_embs = torch.cat(cluster_parts, dim=0)
        graph_emb = torch.cat(graph_parts, dim=0)
    index_table = torch.cat(index_list, dim=0)
    valid_table = torch.cat(valid_list, dim=0)
    demand = torch.cat(demand_list, dim=0)
    customer_to_cluster = torch.cat(c2c_list, dim=0)
    cluster_sizes = torch.cat(size_list, dim=0)
    full_capacity = torch.tensor(cap_list, dtype=torch.float32, device=device)
    base_cluster_logits = cluster_selector(
        cluster_embs, torch.ones(batch, num_clusters, dtype=torch.bool, device=device)
    )

    def rep(x: Tensor) -> Tensor:
        return x.repeat_interleave(num_samples, dim=0)

    routes, cluster_logp, node_logp = _decode_loop(
        node_embs=rep(node_embs),
        base_cluster_logits=rep(base_cluster_logits),
        graph_emb=rep(graph_emb),
        index_table=rep(index_table),
        valid_table=rep(valid_table),
        demand=rep(demand),
        customer_to_cluster=rep(customer_to_cluster),
        cluster_sizes=rep(cluster_sizes),
        full_capacity=rep(full_capacity),
        num_nodes=num_nodes,
        local_pointer=local_pointer,
        reanchor=reanchor,
        mode=mode,
        generator=generator,
        use_checkpoint=use_checkpoint,
        checkpoint_chunk=checkpoint_chunk,
    )

    solutions: list[tuple[Solution, ...]] = []
    costs = torch.zeros(batch, num_samples, dtype=torch.float32)
    for i in range(batch):
        per_instance: list[Solution] = []
        for s in range(num_samples):
            sample_routes = routes[i * num_samples + s]
            solution = Solution(
                routes=tuple(route for route in sample_routes if route),
                metadata={"source": "hierarchical_decode_batched", "mode": mode, "instance": i},
            )
            per_instance.append(solution)
            costs[i, s] = solution_cost(instances[i], solution)
        solutions.append(tuple(per_instance))

    return BatchedRollout(
        solutions=tuple(solutions),
        costs=costs,
        cluster_log_prob=cluster_logp.view(batch, num_samples),
        node_log_prob=node_logp.view(batch, num_samples),
    )


def active_cluster_mask_from_remaining(remaining: Tensor, cluster_index: ClusterIndex) -> Tensor:
    """Active-cluster mask from a global ``remaining`` customer mask."""

    active = [
        remaining[:, torch.as_tensor(cluster, dtype=torch.long, device=remaining.device)].any(dim=1)
        for cluster in cluster_index.clusters
    ]
    return torch.stack(active, dim=1)


def _select_action(
    logits: Tensor,
    mask: Tensor,
    *,
    mode: DecodeMode,
    generator: torch.Generator | None,
) -> Tensor:
    masked = torch.where(mask, logits, logits.new_full((), _NEG_INF))
    if mode == "greedy":
        return masked.argmax(dim=-1)
    if mode == "sample":
        probs = masked.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    raise ValueError(f"unknown decode mode {mode!r}")


def _action_log_prob(logits: Tensor, mask: Tensor, action: Tensor) -> Tensor:
    masked = torch.where(mask, logits, logits.new_full((), _NEG_INF))
    return masked.log_softmax(dim=-1).gather(1, action.unsqueeze(1)).squeeze(1)
