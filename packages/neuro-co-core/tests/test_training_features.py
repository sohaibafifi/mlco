"""EMA wiring, LR scheduler, best-ckpt + resume tests."""

from typing import Any, cast

import pytest
import torch

from neuro_co.core import Trainer
from neuro_co.core.algos.pomo import POMO, POMOConfig
from neuro_co.core.algos.reinforce import REINFORCE, REINFORCEConfig
from neuro_co.core.envs.tsp import TSPEnv
from neuro_co.core.models import AttentionModel
from neuro_co.core.train_utils import build_scheduler


def _model(env):
    return AttentionModel(in_dim=env.encoder_in_dim, hidden_dim=16, num_layers=1, num_heads=2)


def test_build_scheduler_none() -> None:
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    assert build_scheduler(opt, warmup_steps=0, total_steps=0) is None


def test_build_scheduler_cosine() -> None:
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    s = build_scheduler(opt, warmup_steps=2, total_steps=20)
    assert s is not None


def test_pomo_ema_enabled() -> None:
    env = TSPEnv(size=6)
    algo = POMO(
        _model(env), env, POMOConfig(batch_size=4, n_starts=3, eval_batch_size=8, ema_decay=0.9)
    )
    assert algo.ema is not None
    rng = torch.Generator().manual_seed(0)
    algo.train_step(rng)  # ema.update called
    algo.eval_step(rng)  # ema apply/restore, model weights unchanged after
    assert algo.ema._backup is None  # restored


def test_pomo_scheduler_steps_lr() -> None:
    env = TSPEnv(size=6)
    algo = POMO(
        _model(env),
        env,
        POMOConfig(
            batch_size=4,
            n_starts=3,
            eval_batch_size=8,
            lr=1.0,
            lr_warmup_steps=5,
            lr_total_steps=20,
        ),
    )
    assert algo.sched is not None
    rng = torch.Generator().manual_seed(0)
    lr0 = algo.opt.param_groups[0]["lr"]
    algo.train_step(rng)
    lr1 = algo.opt.param_groups[0]["lr"]
    # During warmup, lr increases.
    assert lr1 > lr0


def test_pomo_optimizer_and_weight_decay_are_configured() -> None:
    env = TSPEnv(size=6)
    algo = POMO(
        _model(env),
        env,
        POMOConfig(
            batch_size=4,
            n_starts=3,
            eval_batch_size=8,
            lr=1e-4,
            optimizer="adam",
            weight_decay=1e-6,
        ),
    )
    assert isinstance(algo.opt, torch.optim.Adam)
    assert not isinstance(algo.opt, torch.optim.AdamW)
    assert algo.opt.param_groups[0]["weight_decay"] == 1e-6


def test_reinforce_no_ema_by_default() -> None:
    env = TSPEnv(size=6)
    algo = REINFORCE(_model(env), env, REINFORCEConfig(batch_size=4, eval_batch_size=8))
    assert algo.ema is None
    assert algo.sched is None


def test_optimizer_defaults_preserve_adamw_behavior() -> None:
    env = TSPEnv(size=6)
    pomo = POMO(_model(env), env, POMOConfig(batch_size=4, n_starts=3, eval_batch_size=8))
    reinforce = REINFORCE(_model(env), env, REINFORCEConfig(batch_size=4, eval_batch_size=8))
    for optimizer in (pomo.opt, reinforce.opt):
        assert type(optimizer) is torch.optim.AdamW
        assert optimizer.param_groups[0]["weight_decay"] == 0.01


def test_reinforce_optimizer_and_weight_decay_are_configured() -> None:
    env = TSPEnv(size=6)
    algo = REINFORCE(
        _model(env),
        env,
        REINFORCEConfig(
            batch_size=4,
            eval_batch_size=8,
            lr=1e-4,
            optimizer="adam",
            weight_decay=1e-6,
        ),
    )
    assert isinstance(algo.opt, torch.optim.Adam)
    assert not isinstance(algo.opt, torch.optim.AdamW)
    assert algo.opt.param_groups[0]["weight_decay"] == 1e-6


@pytest.mark.parametrize(
    ("algo_cls", "config"),
    [
        (
            POMO,
            POMOConfig(
                batch_size=4,
                n_starts=3,
                eval_batch_size=8,
                optimizer=cast(Any, "invalid"),
            ),
        ),
        (
            REINFORCE,
            REINFORCEConfig(
                batch_size=4,
                eval_batch_size=8,
                optimizer=cast(Any, "invalid"),
            ),
        ),
    ],
)
def test_optimizer_name_is_validated(algo_cls, config) -> None:
    env = TSPEnv(size=6)
    with pytest.raises(ValueError, match="unsupported optimizer"):
        algo_cls(_model(env), env, config)


def test_eval_is_deterministic_fixed_set() -> None:
    """Fixed seeded eval set: two eval calls (weights unchanged) match exactly."""
    env = TSPEnv(size=6)
    algo = REINFORCE(_model(env), env, REINFORCEConfig(batch_size=4, eval_batch_size=16))
    rng = torch.Generator().manual_seed(0)
    a = algo.eval_step(rng)["eval_tour_length"]
    b = algo.eval_step(rng)["eval_tour_length"]
    assert a == b  # same instances, same weights -> identical


def test_eval_state_cached() -> None:
    env = TSPEnv(size=6)
    algo = POMO(_model(env), env, POMOConfig(batch_size=4, n_starts=3, eval_batch_size=8))
    rng = torch.Generator().manual_seed(0)
    algo.eval_step(rng)
    s1 = algo._eval_state
    algo.eval_step(rng)
    assert algo._eval_state is s1  # not rebuilt


def test_eval_weights_ctx_restores() -> None:
    env = TSPEnv(size=6)
    algo = POMO(
        _model(env), env, POMOConfig(batch_size=4, n_starts=3, eval_batch_size=8, ema_decay=0.9)
    )
    before = {n: p.detach().clone() for n, p in algo.model.named_parameters()}
    with algo.eval_weights():
        pass  # apply + restore
    for n, p in algo.model.named_parameters():
        assert torch.equal(p, before[n])  # restored to training weights


def test_best_ckpt_uses_ema_weights(tmp_path) -> None:
    """best.pt (eval_weights=True) saves EMA shadow, not raw model params.

    Tests the save mechanism directly to avoid best-timing flakiness."""
    env = TSPEnv(size=6)
    algo = REINFORCE(
        _model(env),
        env,
        REINFORCEConfig(batch_size=4, eval_batch_size=8, ema_decay=0.5),
    )
    assert algo.ema is not None
    rng = torch.Generator().manual_seed(0)
    for _ in range(3):
        algo.train_step(rng)  # shadow now diverges from raw model

    sd_key = "model.encoder.embed.weight"
    ema_key = "encoder.embed.weight"
    raw = algo.model.encoder.embed.weight.detach().clone()
    shadow = algo.ema.shadow[ema_key]
    assert not torch.allclose(raw, shadow)  # meaningful test: they differ

    trainer = Trainer(algo=algo, steps=0, ckpt_dir=tmp_path, seed=0)
    trainer._save_ckpt(tmp_path / "best.pt", eval_weights=True, mirror_latest=False)
    best = torch.load(tmp_path / "best.pt", weights_only=False)["algo_state"]
    assert torch.allclose(best[sd_key], shadow, atol=1e-6)  # saved EMA, not raw

    # And raw-weights save (latest) keeps the training params.
    trainer._save_ckpt(tmp_path / "raw.pt", eval_weights=False, mirror_latest=False)
    raw_ckpt = torch.load(tmp_path / "raw.pt", weights_only=False)["algo_state"]
    assert torch.allclose(raw_ckpt[sd_key], raw, atol=1e-6)


def test_trainer_best_ckpt_and_resume(tmp_path) -> None:
    env = TSPEnv(size=6)
    algo = REINFORCE(_model(env), env, REINFORCEConfig(batch_size=4, eval_batch_size=8))
    trainer = Trainer(
        algo=algo,
        steps=6,
        eval_every=2,
        ckpt_every=0,
        ckpt_dir=tmp_path,
        seed=0,
        best_metric_key="eval_reward",
    )
    trainer.fit()
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "latest.pt").exists()
    assert trainer.state.best_metric > float("-inf")

    # Resume into a fresh trainer; step counter restored.
    algo2 = REINFORCE(_model(env), env, REINFORCEConfig(batch_size=4, eval_batch_size=8))
    trainer2 = Trainer(
        algo=algo2,
        steps=0,
        ckpt_dir=tmp_path,
        seed=0,
        resume_from=tmp_path / "latest.pt",
    )
    assert trainer2.state.step == trainer.state.step
    assert trainer2.state.best_metric == trainer.state.best_metric
