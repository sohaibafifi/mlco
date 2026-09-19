import numpy as np
import pytest
import torch

from neuro_co.scale.cluster_encoder import ClusterLocalEncoder
from neuro_co.scale.decode import HierarchicalPolicy, NeighborhoodActionSet
from neuro_co.scale.hierarchical import (
    LinearClusterSelector,
    build_cluster_index,
    gather_neighborhood_action_set,
)
from neuro_co.scale.metrics import evaluate_solution
from neuro_co.scale.partition import morton_partition
from neuro_co.scale.types import Partition, VRPInstance

core_models = pytest.importorskip("neuro_co.core.models")


def _instance(seed: int = 0, num_customers: int = 30) -> VRPInstance:
    rng = np.random.default_rng(seed)
    coords = rng.random((num_customers + 1, 2))
    demand = np.zeros(num_customers + 1)
    demand[1:] = rng.integers(1, 10, size=num_customers)
    return VRPInstance(coords=coords, demand=demand, capacity=30.0)


def test_neighborhood_action_set_spans_anchor_and_neighbors() -> None:
    index = build_cluster_index(
        Partition(clusters=((1, 2), (3, 4), (5, 6)), method="unit"), num_nodes=7
    )
    node_embs = torch.arange(7 * 2, dtype=torch.float32).reshape(1, 7, 2)

    # Anchor cluster 1 (middle) with span 1 -> clusters 0, 1, 2 -> all customers.
    middle = gather_neighborhood_action_set(node_embs, index, torch.tensor([1]), neighbor_span=1)
    assert set(middle.global_indices[0].tolist()) == {0, 1, 2, 3, 4, 5, 6}
    assert middle.global_indices.shape[1] == 3 * index.max_cluster_size + 1  # bounded width

    # Anchor cluster 0 (edge) with span 1 -> clusters 0, 1 only; rest is padding.
    edge = gather_neighborhood_action_set(node_embs, index, torch.tensor([0]), neighbor_span=1)
    real = edge.global_indices[0][edge.mask[0]].tolist()
    assert set(real) == {0, 1, 2, 3, 4}  # depot + clusters 0 and 1


def test_neighborhood_action_set_rejects_nonpositive_span() -> None:
    index = build_cluster_index(Partition(clusters=((1, 2), (3, 4)), method="u"), num_nodes=5)
    node_embs = torch.zeros(1, 5, 2)
    with pytest.raises(ValueError, match="neighbor_span"):
        gather_neighborhood_action_set(node_embs, index, torch.tensor([0]), neighbor_span=0)


def _policy(
    neighbor_span: int, hidden_dim: int = 16, *, reanchor: bool = False
) -> HierarchicalPolicy:
    encoder = ClusterLocalEncoder(in_dim=3, hidden_dim=hidden_dim, num_layers=1, num_heads=2)
    selector = LinearClusterSelector(hidden_dim=hidden_dim)
    pointer = core_models.PointerDecoder(hidden_dim=hidden_dim, num_heads=2)
    builder = NeighborhoodActionSet(neighbor_span) if neighbor_span > 0 else None
    return HierarchicalPolicy(encoder, selector, pointer, builder, reanchor=reanchor)


def test_s2_rollout_is_feasible() -> None:
    instance = _instance(seed=3, num_customers=40)
    partition = morton_partition(instance, max_customers=8)
    policy = _policy(neighbor_span=1).eval()

    rollout = policy.rollout(instance, partition, num_samples=1, mode="greedy")
    metrics = evaluate_solution(instance, rollout.best_solution)

    assert metrics.feasible
    assert metrics.missing_customers == 0
    assert metrics.duplicate_customers == 0
    assert metrics.capacity_violations == 0


def test_s2_allows_cross_cluster_routes() -> None:
    # With span >= 1 a route may contain customers from more than one cluster.
    instance = _instance(seed=4, num_customers=40)
    partition = morton_partition(instance, max_customers=8)
    index = build_cluster_index(partition, num_nodes=instance.num_nodes)
    policy = _policy(neighbor_span=1).eval()

    rollout = policy.rollout(
        instance,
        partition,
        num_samples=16,
        mode="sample",
        generator=torch.Generator().manual_seed(0),
    )
    crossed = False
    for solution in rollout.solutions:
        for route in solution.routes:
            clusters = {index.cluster_of(c) for c in route}
            if len(clusters) > 1:
                crossed = True
    assert crossed, "no sampled route crossed a cluster boundary under S2"


def test_s3_reanchor_is_feasible_and_covers_all() -> None:
    instance = _instance(seed=8, num_customers=50)
    partition = morton_partition(instance, max_customers=6)
    policy = _policy(neighbor_span=1, reanchor=True).eval()

    for mode, samples in [("greedy", 1), ("sample", 8)]:
        gen = torch.Generator().manual_seed(0) if mode == "sample" else None
        rollout = policy.rollout(instance, partition, num_samples=samples, mode=mode, generator=gen)
        for solution in rollout.solutions:
            metrics = evaluate_solution(instance, solution)
            assert metrics.feasible
            assert metrics.missing_customers == 0
            assert metrics.duplicate_customers == 0
            assert metrics.capacity_violations == 0


def test_reanchor_flag_defaults_off_and_is_settable() -> None:
    assert _policy(neighbor_span=1).reanchor is False
    assert _policy(neighbor_span=1, reanchor=True).reanchor is True


def test_s3_feasible_under_loose_capacity() -> None:
    # With capacity that never binds, a route can re-anchor across many clusters;
    # coverage and feasibility must still hold (whether it actually extends routes
    # is a trained behaviour, not asserted here).
    rng = np.random.default_rng(101)
    coords = rng.random((41, 2))
    demand = np.zeros(41)
    demand[1:] = rng.integers(1, 3, size=40)
    instance = VRPInstance(coords=coords, demand=demand, capacity=1000.0)  # loose
    partition = morton_partition(instance, max_customers=5)

    rollout = _policy(1, reanchor=True).eval().rollout(instance, partition, mode="greedy")
    metrics = evaluate_solution(instance, rollout.best_solution)
    assert metrics.feasible
    assert metrics.missing_customers == 0
    assert metrics.duplicate_customers == 0


def test_s1_routes_never_cross_clusters() -> None:
    instance = _instance(seed=4, num_customers=40)
    partition = morton_partition(instance, max_customers=8)
    index = build_cluster_index(partition, num_nodes=instance.num_nodes)
    policy = _policy(neighbor_span=0).eval()  # S1

    rollout = policy.rollout(
        instance,
        partition,
        num_samples=16,
        mode="sample",
        generator=torch.Generator().manual_seed(0),
    )
    for solution in rollout.solutions:
        for route in solution.routes:
            clusters = {index.cluster_of(c) for c in route}
            assert len(clusters) == 1, "S1 route must stay inside one cluster"


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_s3_preserves_route_endpoints_across_reanchors_and_route_resets(
    batched: bool, checkpointed: bool
) -> None:
    """Force several reanchors, then a capacity closure, and inspect context.

    Node IDs are used as embeddings so this checks what the pointer actually
    receives, independently of the decoder's representation of route state.
    """

    class IdentityEncoder(torch.nn.Module):
        def forward(self, features):  # type: ignore[no-untyped-def]
            ids = torch.arange(features.shape[1], dtype=features.dtype)
            nodes = ids[None, :, None].expand(features.shape[0], -1, -1)
            return nodes, nodes.mean(dim=1)

    class OrderedSelector(torch.nn.Module):
        def forward(self, clusters, active):  # type: ignore[no-untyped-def]
            return -torch.arange(clusters.shape[1], dtype=clusters.dtype)[None].expand(
                clusters.shape[0], -1
            )

    class RecordingPointer(torch.nn.Module):
        def __init__(self):  # type: ignore[no-untyped-def]
            super().__init__()
            self.contexts: list[list[tuple[int, int]]] = []

        def forward(self, nodes, graph, first, current, mask):  # type: ignore[no-untyped-def]
            if not self.contexts:
                self.contexts = [[] for _ in range(nodes.shape[0])]
            for row in range(nodes.shape[0]):
                if mask[row].any():
                    self.contexts[row].append(
                        (int(nodes[row, first[row], 0]), int(nodes[row, current[row], 0]))
                    )
            ids = nodes[..., 0]
            # Prefer serving customers to closing. The decoder must mask all
            # already-served context tokens, including ones outside the window.
            return torch.where(ids == 0, -100.0, -ids)

    instance = VRPInstance(
        coords=np.column_stack((np.arange(8), np.zeros(8))),
        demand=np.array([0.0] + [1.0] * 7),
        capacity=3.0,
    )
    partition = Partition(clusters=tuple((i,) for i in range(1, 8)), method="unit")
    pointer = RecordingPointer()
    policy = HierarchicalPolicy(
        IdentityEncoder(),
        OrderedSelector(),
        pointer,
        reanchor=True,
        use_checkpoint=checkpointed,
        checkpoint_chunk=2,
    )
    if batched:
        reversed_partition = Partition(clusters=tuple(reversed(partition.clusters)), method="unit")
        result = policy.rollout_batched([instance, instance], [partition, reversed_partition])
        solutions = [solution for row in result.solutions for solution in row]
    else:
        solutions = list(policy.rollout(instance, partition).solutions)
    assert solutions[0].routes == ((1, 2, 3), (4, 5, 6), (7,))
    if batched:
        assert solutions[1].routes == ((7, 6, 5), (4, 3, 2), (1,))
    for solution in solutions:
        assert evaluate_solution(instance, solution).feasible
    assert pointer.contexts[0] == [
        (0, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (0, 0),
        (4, 4),
        (4, 5),
        (4, 6),
        (0, 0),
        (7, 7),
    ]
    if batched:
        assert pointer.contexts[1] == [
            (0, 0),
            (7, 7),
            (7, 6),
            (7, 5),
            (0, 0),
            (4, 4),
            (4, 3),
            (4, 2),
            (0, 0),
            (1, 1),
        ]
