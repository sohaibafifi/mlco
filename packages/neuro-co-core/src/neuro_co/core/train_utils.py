"""Training utilities: EMA + LR schedulers.

Light helpers, no Lightning. Keep modules independent from `Trainer` so
algos can use them directly without inheriting any framework class.
"""

import copy
import math

import torch
from torch import nn


class WeightEMA:
    """Exponential moving average of model weights.

    Holds a shadow copy of the parameters and updates it after each step.
    Use `apply()` to swap shadow weights into the live model for eval, and
    `restore()` to bring back training weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = decay
        self.model = model
        self.shadow = {
            n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        }
        self._backup: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self) -> None:
        for n, p in self.model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply(self) -> None:
        """Swap shadow into model. Stash the originals for restore()."""
        if self._backup is not None:
            return
        self._backup = {
            n: p.detach().clone() for n, p in self.model.named_parameters() if n in self.shadow
        }
        for n, p in self.model.named_parameters():
            if n in self.shadow:
                p.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self) -> None:
        if self._backup is None:
            return
        for n, p in self.model.named_parameters():
            if n in self._backup:
                p.copy_(self._backup[n])
        self._backup = None


def cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """LR schedule: linear warmup → cosine decay to `min_lr_ratio * base_lr`."""
    if warmup_steps < 0 or total_steps < warmup_steps:
        raise ValueError(f"bad steps: warmup={warmup_steps} total={total_steps}")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def constant_with_warmup(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then constant."""

    def lr_lambda(step: int) -> float:
        return min(1.0, float(step + 1) / float(max(1, warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int = 0,
    total_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Pick a scheduler from simple knobs. None when both are 0.

    - `total_steps > 0`  -> cosine decay with linear warmup
    - else `warmup_steps` -> linear warmup then constant
    - else                -> None (caller skips `.step()`)
    """
    if total_steps > 0:
        return cosine_with_warmup(
            optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=min_lr_ratio
        )
    if warmup_steps > 0:
        return constant_with_warmup(optimizer, warmup_steps=warmup_steps)
    return None


def snapshot(model: nn.Module) -> nn.Module:
    """Deep-copy a model. Helper for baseline networks."""
    cp = copy.deepcopy(model)
    for p in cp.parameters():
        p.requires_grad_(False)
    return cp


__all__ = [
    "WeightEMA",
    "build_scheduler",
    "constant_with_warmup",
    "cosine_with_warmup",
    "snapshot",
]
