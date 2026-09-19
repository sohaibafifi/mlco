import numpy as np
import pytest
import torch

from neuro_co.scale.cluster_encoder import ClusterLocalEncoder
from neuro_co.scale.decode import (
    ClusterAwareEncoder,
    HierarchicalPolicy,
    build_instance_features,
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


def test_cluster_local_encoder_satisfies_cluster_aware_protocol() -> None:
    encoder = ClusterLocalEncoder(in_dim=3, hidden_dim=16, num_layers=2, num_heads=2)

    assert isinstance(encoder, ClusterAwareEncoder)
    assert not isinstance(core_models.AMEncoder(in_dim=3, hidden_dim=16), ClusterAwareEncoder)


def test_encode_clusters_returns_node_cluster_and_graph_embeddings() -> None:
    instance = _instance()
    partition = sweep_partition(instance, max_customers=8)
    cluster_index = build_cluster_index(partition, num_nodes=instance.num_nodes)
    encoder = ClusterLocalEncoder(in_dim=3, hidden_dim=16, num_layers=2, num_heads=2).eval()

    feats = build_instance_features(instance).unsqueeze(0)
    node_embs, cluster_embs, graph_emb = encoder.encode_clusters(feats, cluster_index)

    assert node_embs.shape == (1, instance.num_nodes, 16)
    assert cluster_embs.shape == (1, partition.num_clusters, 16)
    assert graph_emb.shape == (1, 16)
    assert torch.isfinite(node_embs).all()
    assert torch.isfinite(cluster_embs).all()


def test_encode_clusters_rejects_mismatched_node_count() -> None:
    instance = _instance(num_customers=12)
    partition = sweep_partition(instance, max_customers=6)
    cluster_index = build_cluster_index(partition, num_nodes=instance.num_nodes)
    encoder = ClusterLocalEncoder(in_dim=3, hidden_dim=16, num_layers=1, num_heads=2)

    wrong = torch.zeros(1, instance.num_nodes + 3, 3)
    with pytest.raises(ValueError, match="num_nodes"):
        encoder.encode_clusters(wrong, cluster_index)


def _policy(hidden_dim: int = 16) -> HierarchicalPolicy:
    encoder = ClusterLocalEncoder(in_dim=3, hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    selector = LinearClusterSelector(hidden_dim=hidden_dim)
    pointer = core_models.PointerDecoder(hidden_dim=hidden_dim, num_heads=2)
    return HierarchicalPolicy(encoder, selector, pointer).eval()


def test_rollout_with_cluster_local_encoder_is_feasible() -> None:
    instance = _instance(seed=5, num_customers=40)
    partition = morton_partition(instance, max_customers=10)
    policy = _policy()

    rollout = policy.rollout(instance, partition, num_samples=1, mode="greedy")
    metrics = evaluate_solution(instance, rollout.best_solution)

    assert metrics.feasible
    assert metrics.missing_customers == 0
    assert metrics.duplicate_customers == 0
    assert metrics.capacity_violations == 0
    assert torch.isfinite(rollout.log_prob).all()


def test_rollout_with_cluster_local_encoder_is_deterministic_greedy() -> None:
    instance = _instance(seed=6)
    partition = sweep_partition(instance, max_customers=8)
    policy = _policy()

    first = policy.rollout(instance, partition, mode="greedy").best_solution
    second = policy.rollout(instance, partition, mode="greedy").best_solution

    assert first.routes == second.routes


def test_cluster_local_encoder_gradients_flow() -> None:
    instance = _instance(seed=7)
    partition = sweep_partition(instance, max_customers=8)
    policy = _policy()

    rollout = policy.rollout(
        instance,
        partition,
        num_samples=4,
        mode="sample",
        generator=torch.Generator().manual_seed(0),
    )
    # REINFORCE with a group-mean baseline (reward = -cost): below-mean-cost
    # rollouts get their log-prob pushed up.
    advantage = (rollout.costs - rollout.costs.mean()).detach()
    loss = (advantage * rollout.log_prob).mean()
    loss.backward()

    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads, "no gradients reached the policy parameters"
    assert any(torch.isfinite(g).any() and g.abs().sum() > 0 for g in grads)
