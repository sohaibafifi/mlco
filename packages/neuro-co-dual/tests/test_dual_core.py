"""Constraint slack, shaping, and conditioning on core states."""

import torch

from neuro_co.core.models import AttentionModel
from neuro_co.core.trace import rollout_trace
from neuro_co.dual.baselines import shape_reward_scalar
from neuro_co.dual.conditioned import ConstraintConditionedAug, constraint_conditioned_features
from neuro_co.dual.shaping import shape_reward_global
from neuro_co.dual.slack import family_slack, family_violation
from neuro_co.problems.vrptw.env import CVRPTWEnv


def _setup(batch: int = 8):
    torch.manual_seed(0)
    env = CVRPTWEnv(size=6, capacity=20.0, horizon=10.0, window_width=2.0)
    model = AttentionModel(in_dim=env.encoder_in_dim, hidden_dim=16, num_layers=1, num_heads=2)
    state = env.reset(batch, generator=torch.Generator().manual_seed(0))
    return env, model, state


def test_family_slack_cvrptw() -> None:
    env, model, state = _setup()
    tr = rollout_trace(model, env, state)
    slack = family_slack(state, env, tr.actions, "cvrptw")
    assert set(slack) == {"capacity", "time_window", "spatial"}
    for v in slack.values():
        assert v.shape == (8,)
        assert torch.isfinite(v).all()


def test_family_violation() -> None:
    env, model, state = _setup()
    tr = rollout_trace(model, env, state)
    viol = family_violation(state, env, tr.actions, "cvrptw")
    for v in viol.values():
        torch.testing.assert_close(v, torch.zeros_like(v), atol=1e-6, rtol=0)


def test_shape_reward_scalar() -> None:
    cost = torch.zeros(4)
    slack = {"capacity": torch.ones(4), "time_window": torch.full((4,), 0.5)}
    shaped = shape_reward_scalar(cost, slack, alpha=0.1)
    assert torch.allclose(shaped, torch.full((4,), 0.15))


def test_shape_reward_global_fallback() -> None:
    """Global shaping returns finite costs with the available dual backend."""
    env, model, state = _setup(4)
    tr = rollout_trace(model, env, state)
    slack = family_slack(state, env, tr.actions, "cvrptw")
    assert tr.reward is not None
    cost = -tr.reward
    shaped = shape_reward_global(state, env, cost, slack, problem="cvrptw", alpha=0.1)
    assert shaped.shape == (4,)
    assert torch.isfinite(shaped).all()


def test_constraint_conditioned_features() -> None:
    env, _, state = _setup(4)
    feats = constraint_conditioned_features(state, env, "cvrptw")
    assert feats.shape == (4, 3)  # 3 families


def test_conditioned_aug_module() -> None:
    env, _, state = _setup(4)
    aug = ConstraintConditionedAug(d_model=16, n_families=3)
    x = torch.randn(4, 7, 16)
    out = aug(x, state, env, "cvrptw")
    assert out.shape == x.shape
