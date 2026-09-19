import numpy as np
import pytest
import torch

from neuro_co.scale.cluster_encoder import ClusterLocalEncoder
from neuro_co.scale.decode import HierarchicalPolicy, NeighborhoodActionSet
from neuro_co.scale.hierarchical import LinearClusterSelector
from neuro_co.scale.metrics import evaluate_solution
from neuro_co.scale.partition import morton_partition
from neuro_co.scale.types import VRPInstance

core_models = pytest.importorskip("neuro_co.core.models")


def _instance(seed: int, num_customers: int = 30) -> VRPInstance:
    rng = np.random.default_rng(seed)
    coords = rng.random((num_customers + 1, 2))
    demand = np.zeros(num_customers + 1)
    demand[1:] = rng.integers(1, 10, size=num_customers)
    return VRPInstance(coords=coords, demand=demand, capacity=30.0)


def _policy(hidden_dim: int = 16, *, reanchor: bool = False) -> HierarchicalPolicy:
    encoder = ClusterLocalEncoder(in_dim=3, hidden_dim=hidden_dim, num_layers=1, num_heads=2)
    selector = LinearClusterSelector(hidden_dim=hidden_dim)
    pointer = core_models.PointerDecoder(hidden_dim=hidden_dim, num_heads=2)
    return HierarchicalPolicy(encoder, selector, pointer, reanchor=reanchor)


def test_batched_rollout_is_feasible() -> None:
    instances = [_instance(s) for s in range(3)]
    partitions = [morton_partition(i, max_customers=8) for i in instances]
    policy = _policy().eval()

    rollout = policy.rollout_batched(
        instances,
        partitions,
        num_samples=4,
        mode="sample",
        generator=torch.Generator().manual_seed(0),
    )

    assert rollout.costs.shape == (3, 4)
    assert rollout.log_prob.shape == (3, 4)
    for i, instance in enumerate(instances):
        for solution in rollout.solutions[i]:
            assert evaluate_solution(instance, solution).feasible


def test_batched_b1_matches_single_greedy() -> None:
    instance = _instance(7, num_customers=40)
    partition = morton_partition(instance, max_customers=8)

    torch.manual_seed(321)
    policy = _policy().eval()
    single = policy.rollout(instance, partition, num_samples=1, mode="greedy").best_solution
    batched = policy.rollout_batched([instance], [partition], num_samples=1, mode="greedy")

    assert batched.solutions[0][0].routes == single.routes
    assert batched.costs.shape == (1, 1)


def test_batched_requires_equal_sizes() -> None:
    instances = [_instance(0, num_customers=20), _instance(1, num_customers=30)]
    partitions = [morton_partition(i, max_customers=8) for i in instances]
    policy = _policy().eval()

    with pytest.raises(ValueError, match="equal instance sizes"):
        policy.rollout_batched(instances, partitions, num_samples=2, mode="greedy")


def test_batched_logprob_supports_backward() -> None:
    instances = [_instance(s) for s in range(2)]
    partitions = [morton_partition(i, max_customers=8) for i in instances]
    policy = _policy()

    rollout = policy.rollout_batched(
        instances,
        partitions,
        num_samples=4,
        mode="sample",
        generator=torch.Generator().manual_seed(1),
    )
    advantage = (rollout.costs - rollout.costs.mean(dim=1, keepdim=True)).detach()
    loss = (advantage * rollout.log_prob).mean()
    loss.backward()

    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_batched_b1_matches_single_with_reanchor() -> None:
    instance = _instance(9, num_customers=40)
    partition = morton_partition(instance, max_customers=8)

    torch.manual_seed(55)
    policy = _policy(reanchor=True).eval()
    single = policy.rollout(instance, partition, num_samples=1, mode="greedy").best_solution
    batched = policy.rollout_batched([instance], [partition], num_samples=1, mode="greedy")

    assert batched.solutions[0][0].routes == single.routes


@pytest.mark.parametrize("neighbor_span,reanchor", [(0, False), (1, False), (1, True)])
@pytest.mark.parametrize("mode", ["greedy", "sample"])
def test_inference_state_updates_match_autograd_rollout(
    neighbor_span: int, reanchor: bool, mode: str
) -> None:
    instances = [_instance(seed, num_customers=18) for seed in (8, 17)]
    partitions = [morton_partition(instance, max_customers=4) for instance in instances]
    torch.manual_seed(31)
    policy = _policy(reanchor=reanchor).eval()
    if neighbor_span:
        policy.action_set_builder = NeighborhoodActionSet(neighbor_span)
    with torch.enable_grad():
        reference = policy.rollout_batched(
            instances,
            partitions,
            num_samples=3,
            mode=mode,
            generator=torch.Generator().manual_seed(51),
        )
    # Repeated inference also checks that in-place operations cannot contaminate
    # the shared input instances, partitions, or a later rollout.
    for _ in range(2):
        with torch.no_grad():
            inference = policy.rollout_batched(
                instances,
                partitions,
                num_samples=3,
                mode=mode,
                generator=torch.Generator().manual_seed(51),
            )
        for ref_row, row, instance in zip(
            reference.solutions, inference.solutions, instances, strict=True
        ):
            assert [solution.routes for solution in row] == [
                solution.routes for solution in ref_row
            ]
            assert all(evaluate_solution(instance, solution).feasible for solution in row)
        torch.testing.assert_close(inference.costs, reference.costs, rtol=0, atol=0)
        torch.testing.assert_close(inference.log_prob, reference.log_prob, rtol=1e-6, atol=1e-6)
