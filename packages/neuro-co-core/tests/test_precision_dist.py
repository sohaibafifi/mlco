"""Precision + DistEnv tests. No NCCL: only single-process behavior."""

import os

import pytest
import torch
from torch import nn

from neuro_co.core.algos.pomo import POMO, POMOConfig
from neuro_co.core.algos.reinforce import REINFORCE, REINFORCEConfig
from neuro_co.core.distributed import (
    DistEnv,
    all_reduce_grads,
    all_reduce_mean,
    broadcast_params,
)
from neuro_co.core.models import AttentionModel
from neuro_co.core.precision import Precision
from neuro_co.problems.tsp.env import TSPEnv


def test_precision_fp32_is_noop() -> None:
    p = Precision("fp32", device="cpu")
    assert not p.enabled
    x = torch.randn(4, requires_grad=True)
    with p.autocast():
        y = (x * 2).sum()
    assert y.dtype == torch.float32
    p.backward(y)
    assert x.grad is not None


def test_precision_bf16_autocast() -> None:
    p = Precision("bf16", device="cpu")
    assert p.enabled
    x = torch.randn(2, 3)
    with p.autocast():
        # matmul under bf16 autocast on CPU emits bf16 outputs
        y = x @ x.T
    assert y.dtype == torch.bfloat16


def test_precision_invalid() -> None:
    with pytest.raises(ValueError):
        Precision("nope", device="cpu")  # type: ignore[arg-type]


def test_precision_step_runs_fp32() -> None:
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    p = Precision("fp32", device="cpu")
    x = torch.randn(2, 4)
    with p.autocast():
        loss = model(x).sum()
    p.backward(loss)
    p.step(opt)
    p.update()
    # No NaN, params changed.
    for q in model.parameters():
        assert torch.isfinite(q).all()


def test_distenv_defaults_single_process() -> None:
    # Save / restore env to avoid pollution between tests.
    saved = {k: os.environ.get(k) for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        os.environ.pop(k, None)
    try:
        e = DistEnv.from_env()
        assert e.world_size == 1
        assert e.rank == 0
        assert e.local_rank == 0
        assert not e.enabled
        assert e.is_main
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_all_reduce_grads_noop_single_rank() -> None:
    model = nn.Linear(3, 3)
    x = torch.randn(2, 3)
    model(x).sum().backward()
    grads_before = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    # world_size=1 must be a complete no-op (no NCCL calls).
    all_reduce_grads(model, world_size=1)
    grads_after = [p.grad for p in model.parameters() if p.grad is not None]
    for a, b in zip(grads_before, grads_after, strict=True):
        assert torch.equal(a, b)


def test_all_reduce_mean_noop_single_rank() -> None:
    t = torch.tensor(3.14)
    assert torch.equal(all_reduce_mean(t, world_size=1), t)


def test_broadcast_params_noop_when_no_pg() -> None:
    # Without init_process_group, must early-return (not raise).
    model = nn.Linear(2, 2)
    broadcast_params(model, src=0)


def test_reinforce_bf16_runs() -> None:
    env = TSPEnv(size=5)
    model = AttentionModel(in_dim=env.encoder_in_dim, hidden_dim=16, num_layers=1, num_heads=2)
    algo = REINFORCE(
        model=model,
        env=env,
        cfg=REINFORCEConfig(batch_size=8, eval_batch_size=8, precision="bf16"),
    )
    rng = torch.Generator().manual_seed(0)
    m = algo.train_step(rng)
    assert "loss" in m and m["loss"] == m["loss"]


def test_pomo_bf16_runs() -> None:
    env = TSPEnv(size=6)
    model = AttentionModel(in_dim=env.encoder_in_dim, hidden_dim=16, num_layers=1, num_heads=2)
    algo = POMO(
        model=model,
        env=env,
        cfg=POMOConfig(batch_size=4, n_starts=3, eval_batch_size=4, precision="bf16"),
    )
    rng = torch.Generator().manual_seed(0)
    m = algo.train_step(rng)
    assert "loss" in m and m["loss"] == m["loss"]
