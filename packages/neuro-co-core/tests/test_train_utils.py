"""EMA + LR scheduler tests."""

import torch
from torch import nn

from neuro_co.core.train_utils import (
    WeightEMA,
    constant_with_warmup,
    cosine_with_warmup,
    snapshot,
)


def test_ema_update_shifts_shadow() -> None:
    model = nn.Linear(4, 4)
    ema = WeightEMA(model, decay=0.5)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(7.0)
    ema.update()
    # Shadow should be 0.5 * before + 0.5 * 7.0 element-wise.
    for n, p_new in model.named_parameters():
        if n in ema.shadow:
            expected = 0.5 * before[n] + 0.5 * p_new.detach()
            assert torch.allclose(ema.shadow[n], expected)


def test_ema_apply_restore() -> None:
    model = nn.Linear(4, 4)
    ema = WeightEMA(model, decay=0.9)
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(5.0)
    ema.update()  # shadow now mostly original
    ema.apply()
    # Model weights should now be the shadow (mostly original).
    for n, p in model.named_parameters():
        if n in ema.shadow:
            assert torch.allclose(p, ema.shadow[n])
    ema.restore()
    # Model weights should be the post-fill values (5.0).
    for n, p in model.named_parameters():
        if n in original:
            assert torch.allclose(p, torch.full_like(p, 5.0))


def test_cosine_warmup_curve() -> None:
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    sched = cosine_with_warmup(opt, warmup_steps=10, total_steps=100, min_lr_ratio=0.1)
    lrs = []
    for _ in range(100):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    # Warmup phase increases.
    assert lrs[0] < lrs[5] < lrs[9]
    # After warmup, lr <= base.
    assert lrs[9] <= 1.0 + 1e-6
    # End approaches min_lr_ratio.
    assert lrs[-1] < 0.5


def test_constant_warmup_caps() -> None:
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    sched = constant_with_warmup(opt, warmup_steps=5)
    lrs = []
    for _ in range(20):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[0] < 1.0
    assert all(abs(lr - 1.0) < 1e-6 for lr in lrs[10:])


def test_snapshot_freezes() -> None:
    model = nn.Linear(3, 3)
    cp = snapshot(model)
    for p in cp.parameters():
        assert not p.requires_grad
    # Modifying original should not affect copy.
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(99.0)
    for p in cp.parameters():
        assert not (p == 99.0).all()
