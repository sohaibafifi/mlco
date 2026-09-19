"""PPO smoke tests."""

import torch

from neuro_co.core.algos.ppo import PPO, PPOConfig
from neuro_co.core.envs.tsp import TSPEnv
from neuro_co.core.models import AttentionModel


def _build(size: int = 5, batch: int = 16):
    env = TSPEnv(size=size)
    model = AttentionModel(hidden_dim=16, num_layers=1, num_heads=2)
    algo = PPO(
        model=model,
        env=env,
        cfg=PPOConfig(
            batch_size=batch,
            epochs_per_rollout=2,
            eval_batch_size=batch,
            hidden_dim=16,
        ),
    )
    return algo


def test_train_step_runs() -> None:
    algo = _build()
    rng = torch.Generator().manual_seed(0)
    m = algo.train_step(rng)
    for k in ("loss", "policy_loss", "value_loss", "entropy", "ratio_mean", "reward"):
        assert k in m
        assert m[k] == m[k]


def test_eval_step_runs() -> None:
    algo = _build()
    rng = torch.Generator().manual_seed(0)
    m = algo.eval_step(rng)
    assert m["eval_tour_length"] > 0


def test_short_training_loop() -> None:
    algo = _build(size=6, batch=16)
    rng = torch.Generator().manual_seed(0)
    initial = algo.eval_step(rng)["eval_tour_length"]
    for _ in range(3):
        algo.train_step(rng)
    final = algo.eval_step(rng)["eval_tour_length"]
    assert final > 0
    assert final < initial * 3  # not blown up
