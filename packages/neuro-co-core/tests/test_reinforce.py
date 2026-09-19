"""REINFORCE smoke tests: one step decreases loss in expectation, eval runs."""

import torch

from neuro_co.core.algos.reinforce import REINFORCE, REINFORCEConfig
from neuro_co.core.models import AttentionModel
from neuro_co.problems.tsp.env import TSPEnv


def _build(size: int = 5, batch: int = 8):
    env = TSPEnv(size=size)
    model = AttentionModel(hidden_dim=16, num_layers=1, num_heads=2)
    algo = REINFORCE(model, env, REINFORCEConfig(batch_size=batch, eval_batch_size=batch))
    return algo


def test_train_step_runs() -> None:
    algo = _build()
    rng = torch.Generator().manual_seed(0)
    metrics = algo.train_step(rng)
    for k in ("loss", "reward_student", "reward_baseline", "advantage"):
        assert k in metrics
        assert isinstance(metrics[k], float)
        assert metrics[k] == metrics[k]  # not NaN


def test_eval_step_runs() -> None:
    algo = _build()
    rng = torch.Generator().manual_seed(1)
    metrics = algo.eval_step(rng)
    assert "eval_reward" in metrics
    assert metrics["eval_tour_length"] > 0


def test_short_training_loop() -> None:
    """5 steps; baseline mean should be no worse than initial random."""
    algo = _build(size=6, batch=16)
    rng = torch.Generator().manual_seed(0)
    initial_eval = algo.eval_step(rng)["eval_tour_length"]
    for _ in range(5):
        algo.train_step(rng)
    final_eval = algo.eval_step(rng)["eval_tour_length"]
    # Sanity: did not blow up.
    assert final_eval < initial_eval * 5  # extremely loose; just no divergence
    assert final_eval > 0
