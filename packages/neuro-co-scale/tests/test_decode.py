import numpy as np
import pytest
import torch

from neuro_co.scale.cluster_encoder import ClusterLocalEncoder
from neuro_co.scale.decode import (
    HierarchicalPolicy,
    build_instance_features,
    hierarchical_rollout,
)
from neuro_co.scale.hierarchical import LinearClusterSelector, build_cluster_index
from neuro_co.scale.metrics import evaluate_solution
from neuro_co.scale.partition import morton_partition, sweep_partition
from neuro_co.scale.types import VRPInstance

core_models = pytest.importorskip("neuro_co.core.models")


def _instance(seed: int = 0, num_customers: int = 24) -> VRPInstance:
    rng = np.random.default_rng(seed)
    coords = rng.random((num_customers + 1, 2))
    demand = np.zeros(num_customers + 1)
    demand[1:] = rng.integers(1, 10, size=num_customers)
    return VRPInstance(coords=coords, demand=demand, capacity=30.0, name=f"unit{seed}")


def _policy(hidden_dim: int = 16) -> HierarchicalPolicy:
    encoder = core_models.AMEncoder(in_dim=3, hidden_dim=hidden_dim, num_layers=1, num_heads=2)
    selector = LinearClusterSelector(hidden_dim=hidden_dim)
    pointer = core_models.PointerDecoder(hidden_dim=hidden_dim, num_heads=2)
    return HierarchicalPolicy(encoder, selector, pointer).eval()


def test_build_instance_features_shape_and_normalization() -> None:
    instance = _instance()
    feats = build_instance_features(instance)

    assert feats.shape == (instance.num_nodes, 3)
    assert torch.allclose(feats[0, 2], torch.tensor(0.0))  # depot demand is zero
    assert float(feats[:, 2].max()) <= 1.0  # demand normalized by capacity


def test_greedy_rollout_is_feasible_and_serves_every_customer() -> None:
    instance = _instance()
    partition = sweep_partition(instance, max_customers=8)
    policy = _policy()

    rollout = policy.rollout(instance, partition, num_samples=1, mode="greedy")
    metrics = evaluate_solution(instance, rollout.best_solution)

    assert metrics.feasible
    assert metrics.missing_customers == 0
    assert metrics.duplicate_customers == 0
    assert metrics.capacity_violations == 0
    assert rollout.costs.shape == (1,)
    assert torch.isfinite(rollout.log_prob).all()


def test_rollout_matches_golden_routes_s1_and_s2() -> None:
    # Golden values captured from the per-step rollout before the sync-free
    # rewrite. The decode decisions must be byte-for-byte preserved.
    from neuro_co.scale.decode import NeighborhoodActionSet
    from neuro_co.scale.metrics import solution_cost

    rng = np.random.default_rng(7)
    coords = rng.random((61, 2))
    demand = np.zeros(61)
    demand[1:] = rng.integers(1, 10, size=60)
    instance = VRPInstance(coords=coords, demand=demand, capacity=30.0)
    partition = morton_partition(instance, max_customers=8)

    cases = [
        (None, 37, 53.836888, ((45,), (37,), (57,)), (19,)),
        (
            NeighborhoodActionSet(1),
            36,
            53.585171,
            ((40, 56, 47, 25, 23, 45), (51, 46), (54,)),
            (19,),
        ),
    ]
    for builder, n_routes, cost, first3, last in cases:
        torch.manual_seed(123)
        encoder = ClusterLocalEncoder(in_dim=3, hidden_dim=16, num_layers=1, num_heads=2)
        selector = LinearClusterSelector(hidden_dim=16)
        pointer = core_models.PointerDecoder(hidden_dim=16, num_heads=2)
        policy = HierarchicalPolicy(encoder, selector, pointer, builder).eval()

        solution = policy.rollout(instance, partition, num_samples=1, mode="greedy").best_solution

        assert solution.num_routes == n_routes
        assert solution_cost(instance, solution) == pytest.approx(cost, abs=1e-4)
        assert solution.routes[:3] == first3
        assert solution.routes[-1] == last


def test_rollout_is_invariant_to_coordinate_scale() -> None:
    # CVRPLIB uses a [0,1000] integer grid while training uses [0,1]; features are
    # min-max normalized, so the same instance rescaled must decode identically.
    instance = _instance(seed=11)
    big = VRPInstance(
        coords=instance.coords * 1000.0, demand=instance.demand, capacity=instance.capacity
    )
    torch.manual_seed(5)
    policy = _policy().eval()

    small_routes = policy.rollout(
        instance, morton_partition(instance, max_customers=8), mode="greedy"
    ).best_solution.routes
    big_routes = policy.rollout(
        big, morton_partition(big, max_customers=8), mode="greedy"
    ).best_solution.routes

    assert small_routes == big_routes


def test_greedy_rollout_is_deterministic() -> None:
    instance = _instance(seed=1)
    partition = sweep_partition(instance, max_customers=8)
    policy = _policy()

    first = policy.rollout(instance, partition, mode="greedy").best_solution
    second = policy.rollout(instance, partition, mode="greedy").best_solution

    assert first.routes == second.routes


def test_sampled_rollout_returns_finite_logprobs_and_feasible_samples() -> None:
    instance = _instance(seed=2)
    partition = sweep_partition(instance, max_customers=8)
    policy = _policy()
    generator = torch.Generator().manual_seed(123)

    rollout = policy.rollout(instance, partition, num_samples=8, mode="sample", generator=generator)

    assert rollout.costs.shape == (8,)
    assert rollout.cluster_log_prob.shape == (8,)
    assert rollout.node_log_prob.shape == (8,)
    assert torch.isfinite(rollout.log_prob).all()
    assert (rollout.log_prob <= 1e-4).all()  # log-probabilities are non-positive
    for solution in rollout.solutions:
        assert evaluate_solution(instance, solution).feasible
    assert rollout.best_cost == pytest.approx(float(rollout.costs.min()))


def test_local_action_set_is_bounded_by_cluster_size() -> None:
    instance = _instance(seed=3)
    partition = sweep_partition(instance, max_customers=6)
    policy = _policy()

    seen_widths: list[int] = []
    pointer = policy.local_pointer

    def spy(node_embs, graph_emb, first_idx, current_idx, mask):  # type: ignore[no-untyped-def]
        seen_widths.append(node_embs.shape[1])
        return pointer(node_embs, graph_emb, first_idx, current_idx, mask)

    cluster_index = build_cluster_index(partition, num_nodes=instance.num_nodes)
    hierarchical_rollout(
        instance,
        cluster_index,
        encoder=policy.encoder,
        cluster_selector=policy.cluster_selector,
        local_pointer=spy,
        mode="greedy",
    )

    # action set = depot + at most max_cluster_size customers, never the whole instance
    max_cluster_size = max(len(cluster) for cluster in partition.clusters)
    assert seen_widths, "pointer was never called"
    assert max(seen_widths) == max_cluster_size + 1
    assert max(seen_widths) < instance.num_nodes


def test_rollout_rejects_demand_exceeding_capacity() -> None:
    instance = _instance(seed=4)
    partition = sweep_partition(instance, max_customers=8)
    policy = _policy()
    infeasible = VRPInstance(
        coords=instance.coords,
        demand=np.concatenate([[0.0], np.full(instance.num_customers, 100.0)]),
        capacity=30.0,
    )

    cluster_index = build_cluster_index(partition, num_nodes=infeasible.num_nodes)
    with pytest.raises(ValueError, match="capacity"):
        hierarchical_rollout(
            infeasible,
            cluster_index,
            encoder=policy.encoder,
            cluster_selector=policy.cluster_selector,
            local_pointer=policy.local_pointer,
        )
