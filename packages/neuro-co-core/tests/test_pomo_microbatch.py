"""Microbatches preserve one logical POMO update and instance weighting."""

from unittest.mock import Mock

import pytest
import torch

from neuro_co.core.algos.pomo import POMO, POMOConfig
from neuro_co.core.models import AttentionModel
from neuro_co.problems.cvrp.env import CVRPEnv
from neuro_co.problems.tsp.env import TSPEnv


def _algo(env=None, **settings):
    env = env or TSPEnv(size=5)
    model = AttentionModel(in_dim=env.encoder_in_dim, hidden_dim=16, num_layers=1, num_heads=2)
    return POMO(model, env, POMOConfig(batch_size=5, n_starts=2, **settings))


@pytest.mark.parametrize("microbatch", [None, 1, 3, 5, 8])
@pytest.mark.parametrize("problem", ["tsp", "cvrp"])
def test_real_pomo_training(microbatch, problem):
    env = TSPEnv(size=5) if problem == "tsp" else CVRPEnv(size=5, capacity=20.0)
    algo = _algo(env, microbatch_size=microbatch)
    result = algo.train_step(torch.Generator().manual_seed(2))
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())
    assert algo._step == 1
    assert algo.cfg.batch_size == 5
    assert any(parameter.grad is not None for parameter in algo.model.parameters())


@pytest.mark.parametrize("microbatch", [None, 2, 100])
def test_weighted_gradients_and_metrics_with_supplied_state(monkeypatch, microbatch):
    """An uneven final chunk must receive its fraction of the logical loss."""
    algo = _algo(microbatch_size=microbatch, grad_clip=0)
    algo.cfg.batch_size = 9  # The injected state determines the actual logical batch.
    state = algo.env.reset(5, generator=torch.Generator().manual_seed(3))
    values = torch.arange(1.0, 6.0)
    state = state.replace(coords=state.coords.clone())
    state.coords[:, 0, 0] = values
    parameter = next(algo.model.parameters())
    initial = parameter.detach().clone()
    algo.opt = torch.optim.SGD(algo.model.parameters(), lr=0.1)
    seen = []

    def rollout(*, rng, sample_actions, state):
        assert sample_actions
        x = state.coords[:, 0, 0]
        seen.append(x.clone())
        reward = torch.stack((x, -2 * x), dim=1).reshape(-1)
        sum_logp = torch.stack((parameter.flatten()[0] * x, 2 * parameter.flatten()[0] * x), dim=1)
        return reward, sum_logp.reshape(-1)

    monkeypatch.setattr(algo, "_pomo_rollout", rollout)
    monkeypatch.setattr(algo.env, "reset", Mock(side_effect=AssertionError("unexpected reset")))
    result = algo.train_step(torch.Generator().manual_seed(2), state=state)

    # adv = (+1.5*x, -1.5*x), so loss = 0.75 * parameter * mean(x**2).
    derivative = 0.75 * values.square().mean()
    assert parameter.flatten()[0].item() == pytest.approx(
        (initial.flatten()[0] - 0.1 * derivative).item(), abs=1e-6
    )
    assert torch.equal(torch.cat(seen), values)
    assert [len(chunk) for chunk in seen] == ([2, 2, 1] if microbatch == 2 else [5])
    assert result["loss"] == pytest.approx((initial.flatten()[0] * derivative).item(), abs=1e-6)
    assert result["reward_mean"] == pytest.approx(-values.mean().item() / 2)
    assert result["reward_max_per_group"] == pytest.approx(values.mean().item())
    assert result["advantage_std"] == pytest.approx((2.25 * values.square().mean()).sqrt().item())
    assert algo.cfg.batch_size == 9


def test_one_optimizer_scheduler_ema_and_scaler_update(monkeypatch):
    algo = _algo(microbatch_size=2, ema_decay=0.9, lr_total_steps=10)
    names = ["backward", "unscale_", "step", "update"]
    calls = {}
    for name in names:
        calls[name] = Mock(wraps=getattr(algo.precision, name))
        monkeypatch.setattr(algo.precision, name, calls[name])
    scheduler_step = Mock(wraps=algo.sched.step)
    ema_update = Mock(wraps=algo.ema.update)
    monkeypatch.setattr(algo.sched, "step", scheduler_step)
    monkeypatch.setattr(algo.ema, "update", ema_update)
    algo.train_step(torch.Generator().manual_seed(2))
    assert calls["backward"].call_count == 3
    for name in ("unscale_", "step", "update"):
        assert calls[name].call_count == 1
    assert scheduler_step.call_count == 1
    assert ema_update.call_count == 1
    assert algo._step == 1
    assert all(state["step"].item() == 1 for state in algo.opt.state.values())


@pytest.mark.parametrize("microbatch", [0, -1, 1.5, True])
def test_invalid_microbatch_size(microbatch):
    with pytest.raises(ValueError, match="microbatch_size"):
        _algo(microbatch_size=microbatch)


def test_empty_supplied_batch_rejected():
    algo = _algo(microbatch_size=2)
    with pytest.raises(ValueError, match="at least one instance"):
        algo.train_step(torch.Generator(), state=algo.env.reset(0))
