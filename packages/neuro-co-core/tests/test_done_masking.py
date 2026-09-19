"""Regression: variable-length rollouts must mask finished episodes.

Bug (fixed): CVRP/CVRPTW/OP/PDP/mTSP finish at different steps and the env
re-emits reward on the absorbing terminal state. Rollouts that kept
accumulating reward + log-prob after `done` double-counted returns and
corrupted the gradient. Symptom: eval tour length inflated ~3x, POMO/
REINFORCE stuck. TSP/ATSP (fixed length) were unaffected.

Test idea: padding the rollout with extra steps must NOT change the
returned reward: finished episodes contribute nothing past `done`.
"""

import torch

from neuro_co.core.algos.pomo import _greedy_rollout
from neuro_co.core.algos.reinforce import _rollout
from neuro_co.core.envs.cvrp import CVRPEnv
from neuro_co.core.envs.op import OPEnv
from neuro_co.core.models import AttentionModel


class _Padded:
    """Wrap an env, inflating max_steps to force post-done iterations."""

    def __init__(self, env, factor: int) -> None:
        self._env = env
        self._factor = factor
        self.encoder_in_dim = env.encoder_in_dim

    def __getattr__(self, name):
        return getattr(self._env, name)

    def max_steps(self, state) -> int:
        return self._env.max_steps(state) * self._factor


def _model(env):
    return AttentionModel(
        in_dim=env.encoder_in_dim, hidden_dim=16, num_layers=1, num_heads=2
    ).eval()


def test_reinforce_rollout_invariant_to_padding_cvrp() -> None:
    env = CVRPEnv(size=8, capacity=20.0)
    model = _model(env)
    g = torch.Generator().manual_seed(0)
    state = env.reset(32, generator=g, device="cpu")
    with torch.no_grad():
        r_norm, _ = _rollout(model, env, state, sample_actions=False, rng=None)
        r_pad, _ = _rollout(model, _Padded(env, 3), state, sample_actions=False, rng=None)
    assert torch.allclose(r_norm, r_pad, atol=1e-5)


def test_greedy_rollout_invariant_to_padding_op() -> None:
    env = OPEnv(size=8, budget=4.0)
    model = _model(env)
    g = torch.Generator().manual_seed(0)
    state = env.reset(32, generator=g, device="cpu")
    with torch.no_grad():
        r_norm = _greedy_rollout(model, env, state)
        r_pad = _greedy_rollout(model, _Padded(env, 3), state)
    assert torch.allclose(r_norm, r_pad, atol=1e-5)


def test_cvrp_eval_not_inflated() -> None:
    """Greedy CVRP tour length must be in a sane range, not 3x inflated."""
    env = CVRPEnv(size=10, capacity=20.0)
    model = _model(env)
    state = env.reset(128, generator=torch.Generator().manual_seed(0), device="cpu")
    with torch.no_grad():
        reward = _greedy_rollout(model, env, state)
    tour = -reward.mean().item()
    # Untrained upper bound: depot-after-every-customer ~ 2*n*avg_depot_dist.
    # For n=10 that's well under 25; pre-fix inflation pushed this far higher.
    assert 0 < tour < 25, tour
