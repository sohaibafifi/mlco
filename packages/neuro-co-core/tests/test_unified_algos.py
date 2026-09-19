"""Smoke tests proving REINFORCE / POMO / PPO work across all envs via Env Protocol.

No per-env algo variants: same algo class handles TSP, CVRP, CVRPTW
because Env Protocol abstracts feature building, decoder context, and
rollout bounds.
"""

import pytest
import torch

from neuro_co.core.algos.pomo import POMO, POMOConfig
from neuro_co.core.algos.ppo import PPO, PPOConfig
from neuro_co.core.algos.reinforce import REINFORCE, REINFORCEConfig
from neuro_co.core.env import Env
from neuro_co.core.envs.cvrp import CVRPEnv
from neuro_co.core.envs.cvrptw import CVRPTWEnv
from neuro_co.core.envs.fjsp import FJSPEnv
from neuro_co.core.envs.mtsp import MTSPEnv
from neuro_co.core.envs.op import OPEnv
from neuro_co.core.envs.pdp import PDPEnv
from neuro_co.core.envs.tsp import TSPEnv
from neuro_co.core.models import AttentionModel

ENVS = [
    ("tsp", lambda: TSPEnv(size=6)),
    ("cvrp", lambda: CVRPEnv(size=6, capacity=20.0)),
    ("cvrptw", lambda: CVRPTWEnv(size=6, capacity=20.0, horizon=10.0, window_width=2.0)),
    ("op", lambda: OPEnv(size=6, budget=4.0)),
    ("pdp", lambda: PDPEnv(size=3)),
    ("mtsp", lambda: MTSPEnv(size=6, num_agents=2)),
    ("fjsp", lambda: FJSPEnv(size=3, ops_per_job=2, num_machines=4)),
]


def _model(env: Env):
    return AttentionModel(in_dim=env.encoder_in_dim, hidden_dim=16, num_layers=1, num_heads=2)


@pytest.mark.parametrize(("name", "make_env"), ENVS)
def test_env_satisfies_protocol(name: str, make_env) -> None:
    env = make_env()
    assert isinstance(env, Env)


@pytest.mark.parametrize(("name", "make_env"), ENVS)
def test_reinforce_works(name: str, make_env) -> None:
    env = make_env()
    algo = REINFORCE(
        model=_model(env),
        env=env,
        cfg=REINFORCEConfig(batch_size=8, eval_batch_size=8),
    )
    rng = torch.Generator().manual_seed(0)
    m = algo.train_step(rng)
    assert "loss" in m and m["loss"] == m["loss"]
    ev = algo.eval_step(rng)
    # Metrics finite. Sign varies by problem: OP reward is +prize (so
    # eval_tour_length = -prize < 0); routing problems are -length (> 0).
    assert ev["eval_tour_length"] == ev["eval_tour_length"]  # not NaN
    assert ev["eval_reward"] == ev["eval_reward"]


@pytest.mark.parametrize(("name", "make_env"), ENVS)
def test_pomo_works(name: str, make_env) -> None:
    env = make_env()
    algo = POMO(
        model=_model(env),
        env=env,
        cfg=POMOConfig(batch_size=8, n_starts=3, eval_batch_size=8),
    )
    rng = torch.Generator().manual_seed(0)
    m = algo.train_step(rng)
    assert "loss" in m and m["loss"] == m["loss"]
    ev = algo.eval_step(rng)
    # Metrics finite. Sign varies by problem: OP reward is +prize (so
    # eval_tour_length = -prize < 0); routing problems are -length (> 0).
    assert ev["eval_tour_length"] == ev["eval_tour_length"]  # not NaN
    assert ev["eval_reward"] == ev["eval_reward"]


@pytest.mark.parametrize(("name", "make_env"), ENVS)
def test_ppo_works(name: str, make_env) -> None:
    env = make_env()
    algo = PPO(
        model=_model(env),
        env=env,
        cfg=PPOConfig(batch_size=8, eval_batch_size=8, hidden_dim=16, epochs_per_rollout=1),
    )
    rng = torch.Generator().manual_seed(0)
    m = algo.train_step(rng)
    assert "loss" in m and m["loss"] == m["loss"]
    ev = algo.eval_step(rng)
    # Metrics finite. Sign varies by problem: OP reward is +prize (so
    # eval_tour_length = -prize < 0); routing problems are -length (> 0).
    assert ev["eval_tour_length"] == ev["eval_tour_length"]  # not NaN
    assert ev["eval_reward"] == ev["eval_reward"]
