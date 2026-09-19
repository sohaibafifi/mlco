from copy import deepcopy

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


def _instance(seed: int = 0, num_customers: int = 40) -> VRPInstance:
    rng = np.random.default_rng(seed)
    coords = rng.random((num_customers + 1, 2))
    demand = np.zeros(num_customers + 1)
    demand[1:] = rng.integers(1, 10, size=num_customers)
    return VRPInstance(coords=coords, demand=demand, capacity=30.0)


def _policy(*, reanchor: bool = False, hidden_dim: int = 16) -> HierarchicalPolicy:
    encoder = ClusterLocalEncoder(in_dim=3, hidden_dim=hidden_dim, num_layers=1, num_heads=2)
    selector = LinearClusterSelector(hidden_dim=hidden_dim)
    pointer = core_models.PointerDecoder(hidden_dim=hidden_dim, num_heads=2)
    builder = NeighborhoodActionSet(1) if reanchor else None
    return HierarchicalPolicy(encoder, selector, pointer, builder, reanchor=reanchor)


@pytest.mark.parametrize("reanchor", [False, True])
def test_checkpoint_greedy_matches_plain(reanchor: bool) -> None:
    instance = _instance()
    partition = morton_partition(instance, max_customers=8)
    torch.manual_seed(11)
    policy = _policy(reanchor=reanchor).eval()

    policy.use_checkpoint = False
    plain = policy.rollout(instance, partition, num_samples=1, mode="greedy").best_solution
    policy.use_checkpoint = True
    policy.checkpoint_chunk = 16
    checkpointed = policy.rollout(instance, partition, num_samples=1, mode="greedy").best_solution

    assert checkpointed.routes == plain.routes


def test_checkpoint_sample_is_feasible_and_backprops() -> None:
    instance = _instance(seed=2)
    partition = morton_partition(instance, max_customers=8)
    torch.manual_seed(3)
    policy = _policy()
    policy.use_checkpoint = True
    policy.checkpoint_chunk = 16

    rollout = policy.rollout(
        instance,
        partition,
        num_samples=6,
        mode="sample",
        generator=torch.Generator().manual_seed(0),
    )
    for solution in rollout.solutions:
        assert evaluate_solution(instance, solution).feasible

    advantage = (rollout.costs - rollout.costs.mean()).detach()
    loss = (advantage * rollout.log_prob).mean()
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_checkpoint_chunk_size_invariant_for_greedy() -> None:
    # Greedy result must not depend on the checkpoint chunk size.
    instance = _instance(seed=5, num_customers=50)
    partition = morton_partition(instance, max_customers=8)
    torch.manual_seed(7)
    policy = _policy().eval()
    policy.use_checkpoint = True

    policy.checkpoint_chunk = 8
    a = policy.rollout(instance, partition, mode="greedy").best_solution
    policy.checkpoint_chunk = 64
    b = policy.rollout(instance, partition, mode="greedy").best_solution
    assert a.routes == b.routes


@pytest.mark.parametrize("reanchor", [False, True])
@pytest.mark.parametrize("explicit_generator", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_checkpoint_sampling_matches_plain_routes_logprobs_and_gradients(
    reanchor: bool, explicit_generator: bool, batched: bool
) -> None:
    """Checkpoint recomputation must use the same sampled actions and context."""
    instances = [_instance(seed=s, num_customers=16) for s in (21, 22)]
    partitions = [morton_partition(instance, max_customers=3) for instance in instances]
    torch.manual_seed(17)
    original = _policy(reanchor=reanchor)
    outcomes = []
    for checkpointed in (False, True):
        policy = deepcopy(original)
        policy.use_checkpoint = checkpointed
        policy.checkpoint_chunk = 3
        torch.manual_seed(91)
        generator = torch.Generator().manual_seed(77) if explicit_generator else None
        global_state = torch.random.get_rng_state().clone()
        if batched:
            result = policy.rollout_batched(
                instances, partitions, num_samples=4, mode="sample", generator=generator
            )
            solutions = result.solutions
            advantage = (result.costs - result.costs.mean(dim=1, keepdim=True)).detach()
            for instance, row in zip(instances, solutions, strict=True):
                assert all(evaluate_solution(instance, solution).feasible for solution in row)
        else:
            result = policy.rollout(
                instances[0], partitions[0], num_samples=4, mode="sample", generator=generator
            )
            solutions = (result.solutions,)
            advantage = (result.costs - result.costs.mean()).detach()
            assert all(
                evaluate_solution(instances[0], solution).feasible for solution in result.solutions
            )
        if generator is not None:
            assert torch.equal(global_state, torch.random.get_rng_state())
            generator_after_forward = generator.get_state().clone()
        loss = (advantage * result.log_prob).mean()
        loss.backward()
        if generator is not None:
            assert torch.equal(generator_after_forward, generator.get_state())
        gradients = {
            name: None if param.grad is None else param.grad.detach().clone()
            for name, param in policy.named_parameters()
        }
        assert all(torch.isfinite(grad).all() for grad in gradients.values() if grad is not None)
        assert any(grad.abs().sum() > 0 for grad in gradients.values() if grad is not None)
        outcomes.append(
            (
                tuple(tuple(solution.routes for solution in row) for row in solutions),
                result.log_prob.detach(),
                gradients,
            )
        )
    plain, checkpointed = outcomes
    assert checkpointed[0] == plain[0]
    torch.testing.assert_close(checkpointed[1], plain[1], rtol=1e-6, atol=1e-6)
    for name, grad in plain[2].items():
        other = checkpointed[2][name]
        if grad is None:
            assert other is None
        else:
            assert other is not None
            torch.testing.assert_close(other, grad, rtol=2e-5, atol=2e-5, msg=name)


def test_checkpoint_rejects_nonpositive_chunk_size() -> None:
    instance = _instance(num_customers=8)
    policy = _policy()
    policy.use_checkpoint = True
    policy.checkpoint_chunk = 0
    with pytest.raises(ValueError, match="checkpoint_chunk"):
        policy.rollout(instance, morton_partition(instance, max_customers=3))
