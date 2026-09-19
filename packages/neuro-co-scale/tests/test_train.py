from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from neuro_co.scale.cluster_encoder import ClusterLocalEncoder
from neuro_co.scale.decode import HierarchicalPolicy, NeighborhoodActionSet
from neuro_co.scale.hierarchical import LinearClusterSelector
from neuro_co.scale.partition import morton_partition
from neuro_co.scale.train import (
    TrainConfig,
    _lr_factor,
    reinforce_loss,
    train_hierarchical,
)
from neuro_co.scale.types import VRPInstance

core_models = pytest.importorskip("neuro_co.core.models")


def _instances(step: int) -> list[VRPInstance]:
    rng = np.random.default_rng(step)
    batch = []
    for _ in range(2):
        coords = rng.random((21, 2))
        demand = np.zeros(21)
        demand[1:] = rng.integers(1, 10, size=20)
        batch.append(VRPInstance(coords=coords, demand=demand, capacity=30.0))
    return batch


def _policy(hidden_dim: int = 16) -> HierarchicalPolicy:
    encoder = ClusterLocalEncoder(in_dim=3, hidden_dim=hidden_dim, num_layers=1, num_heads=2)
    selector = LinearClusterSelector(hidden_dim=hidden_dim)
    pointer = core_models.PointerDecoder(hidden_dim=hidden_dim, num_heads=2)
    return HierarchicalPolicy(encoder, selector, pointer)


def test_reinforce_loss_is_scalar_and_finite() -> None:
    costs = torch.tensor([2.0, 4.0, 3.0])
    log_prob = torch.tensor([-1.0, -2.0, -1.5], requires_grad=True)

    loss, baseline = reinforce_loss(costs, log_prob)

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert baseline == pytest.approx(3.0)
    loss.backward()
    assert log_prob.grad is not None


def test_reinforce_loss_requires_a_group() -> None:
    with pytest.raises(ValueError, match="group baseline"):
        reinforce_loss(torch.tensor([2.0]), torch.tensor([-1.0], requires_grad=True))


def test_train_runs_and_updates_parameters() -> None:
    policy = _policy()
    partition_fn = lambda instance: morton_partition(instance, max_customers=7)  # noqa: E731
    before = next(policy.parameters()).detach().clone()

    history = train_hierarchical(
        policy,
        partition_fn=partition_fn,
        instances=_instances,
        cfg=TrainConfig(steps=6, n_starts=4, batch_instances=2, lr=1e-2, seed=0),
    )

    assert len(history) == 6
    assert all(np.isfinite(step.loss) for step in history)
    assert all(np.isfinite(step.mean_cost) for step in history)
    after = next(policy.parameters()).detach()
    assert not torch.allclose(before, after), "training did not update parameters"


def test_lr_factor_warms_up_then_decays() -> None:
    # Linear warmup to 1.0 over 10 steps, then cosine decay toward 0 by step 100.
    assert _lr_factor(0, 10, 100) == pytest.approx(0.1)
    assert _lr_factor(9, 10, 100) == pytest.approx(1.0)
    assert _lr_factor(10, 10, 100) == pytest.approx(1.0)
    assert _lr_factor(99, 10, 100) < 0.01  # near zero at the end
    assert _lr_factor(0, 0, 100) == pytest.approx(1.0)  # no warmup -> cosine from full


def test_train_with_warmup_runs_and_updates() -> None:
    policy = _policy()
    partition_fn = lambda instance: morton_partition(instance, max_customers=7)  # noqa: E731
    before = next(policy.parameters()).detach().clone()

    history = train_hierarchical(
        policy,
        partition_fn=partition_fn,
        instances=_instances,
        cfg=TrainConfig(steps=6, n_starts=4, lr=1e-2, lr_warmup_steps=2, seed=0),
    )

    assert len(history) == 6
    assert all(np.isfinite(step.loss) for step in history)
    assert not torch.allclose(before, next(policy.parameters()).detach())


def test_train_rejects_degenerate_config() -> None:
    policy = _policy()
    partition_fn = lambda instance: morton_partition(instance, max_customers=7)  # noqa: E731

    with pytest.raises(ValueError, match="n_starts"):
        train_hierarchical(
            policy,
            partition_fn=partition_fn,
            instances=_instances,
            cfg=TrainConfig(steps=2, n_starts=1),
        )


def _assert_state_equal(left, right) -> None:  # type: ignore[no-untyped-def]
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.device.type == right.device.type == "cpu"
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_state_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("checkpointed", [False, True])
def test_interrupted_training_resume_is_exact_on_cpu(tmp_path, checkpointed: bool) -> None:  # type: ignore[no-untyped-def]
    def resume_policy() -> HierarchicalPolicy:
        # Encoder dropout exercises global PyTorch RNG restoration in addition
        # to the explicit decoder sampling generator.
        return HierarchicalPolicy(
            ClusterLocalEncoder(in_dim=3, hidden_dim=8, num_layers=1, num_heads=2, dropout=0.15),
            LinearClusterSelector(hidden_dim=8),
            core_models.PointerDecoder(hidden_dim=8, num_heads=2),
            NeighborhoodActionSet(1),
            reanchor=True,
            use_checkpoint=checkpointed,
            checkpoint_chunk=3,
        )

    def tiny_instances(step: int) -> list[VRPInstance]:
        rng = np.random.default_rng(step + 13)
        return [
            VRPInstance(
                coords=rng.random((11, 2)),
                demand=np.r_[0.0, rng.integers(1, 4, 10)],
                capacity=7.0,
            )
        ]

    def on_step(record):  # type: ignore[no-untyped-def]
        # A callback may consume RNG. Snapshot ordering must preserve the state
        # after that callback for the next update.
        torch.rand(3)

    cfg = TrainConfig(steps=5, n_starts=3, batch_instances=1, lr=0.003, lr_warmup_steps=2, seed=57)
    partition_fn = lambda instance: morton_partition(instance, max_customers=3)  # noqa: E731
    torch.manual_seed(731)
    initial = resume_policy()
    initial_rng = torch.random.get_rng_state().clone()
    reference = deepcopy(initial)
    reference_states = []
    reference_history = train_hierarchical(
        reference,
        partition_fn=partition_fn,
        instances=tiny_instances,
        cfg=cfg,
        on_step=on_step,
        on_checkpoint=reference_states.append,
        checkpoint_every=2,
    )
    assert [state["completed_step"] for state in reference_states] == [2, 4, 5]

    interrupted = deepcopy(initial)
    torch.random.set_rng_state(initial_rng)
    prefix_states = []
    prefix_history = train_hierarchical(
        interrupted,
        partition_fn=partition_fn,
        instances=tiny_instances,
        cfg=cfg,
        on_step=on_step,
        on_checkpoint=prefix_states.append,
        checkpoint_every=2,
        max_steps=3,
    )
    assert prefix_history == reference_history[:3]
    assert [state["completed_step"] for state in prefix_states] == [2, 3]
    checkpoint_path = tmp_path / "training_state.pt"
    torch.save(prefix_states[-1], checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    saved_before_resume = deepcopy(loaded)

    # New process-like initialization intentionally disturbs the global RNG and
    # model weights before loading the persisted optimizer/scheduler/RNG state.
    torch.manual_seed(444)
    restored = resume_policy()
    torch.rand(25)
    resumed_states = []
    resumed_history = train_hierarchical(
        restored,
        partition_fn=partition_fn,
        instances=tiny_instances,
        cfg=cfg,
        on_step=on_step,
        resume_state=loaded,
        on_checkpoint=resumed_states.append,
        checkpoint_every=2,
    )
    assert resumed_history == reference_history
    assert [state["completed_step"] for state in resumed_states] == [4, 5]
    # This compares parameters, Adam moments, LR scheduler, sampling stream,
    # global CPU/CUDA RNG snapshots, metadata, and full history exactly.
    _assert_state_equal(resumed_states[-1], reference_states[-1])
    _assert_state_equal(loaded, saved_before_resume)


def test_training_resume_rejects_schedule_or_policy_changes() -> None:
    cfg = TrainConfig(steps=3, n_starts=3, lr_warmup_steps=1)
    states = []
    partition_fn = lambda instance: morton_partition(instance, max_customers=7)  # noqa: E731
    train_hierarchical(
        _policy(),
        partition_fn=partition_fn,
        instances=_instances,
        cfg=cfg,
        on_checkpoint=states.append,
        max_steps=1,
    )
    with pytest.raises(ValueError, match="train_config"):
        train_hierarchical(
            _policy(),
            partition_fn=partition_fn,
            instances=_instances,
            cfg=replace(cfg, steps=4),
            resume_state=states[-1],
        )
    changed_policy = _policy()
    changed_policy.reanchor = True
    with pytest.raises(ValueError, match="policy_config"):
        train_hierarchical(
            changed_policy,
            partition_fn=partition_fn,
            instances=_instances,
            cfg=cfg,
            resume_state=states[-1],
        )


def test_checkpoint_snapshots_do_not_alias_later_updates() -> None:
    states = []
    first_copy = []

    def capture(state):  # type: ignore[no-untyped-def]
        states.append(state)
        if not first_copy:
            first_copy.append(deepcopy(state))

    train_hierarchical(
        _policy(),
        partition_fn=lambda instance: morton_partition(instance, max_customers=7),
        instances=_instances,
        cfg=TrainConfig(steps=3, n_starts=3),
        on_checkpoint=capture,
    )
    assert [state["completed_step"] for state in states] == [1, 2, 3]
    _assert_state_equal(states[0], first_copy[0])
