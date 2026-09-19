"""Precision control: `fp32`, `bf16`, `fp16` via `torch.amp.autocast`.

`Precision` is a tiny wrapper bundling autocast context + optional
`GradScaler`. Algos:
  - `with precision.autocast(): ... loss = ...`
  - `precision.backward(loss)`
  - `precision.step(opt)`
  - `precision.update()`  # no-op without scaler

For `fp32` everything is a no-op. For `bf16` autocast only (no scaler
needed). For `fp16` autocast + GradScaler.
"""

import contextlib
from typing import Literal

import torch
from torch import nn

PrecisionType = Literal["fp32", "bf16", "fp16"]


_DTYPE: dict[str, torch.dtype | None] = {
    "fp32": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


class Precision:
    """Bundle of autocast + scaler. Algo-agnostic."""

    def __init__(self, precision: PrecisionType = "fp32", device: str = "cpu") -> None:
        if precision not in _DTYPE:
            raise ValueError(f"unknown precision: {precision!r}, want fp32|bf16|fp16")
        self.precision: PrecisionType = precision
        self.device = device
        dtype = _DTYPE[precision]
        self._dtype: torch.dtype | None = dtype
        # `GradScaler` is only useful with fp16. bf16 has the dynamic range of
        # fp32 so no scaling needed; fp32 obviously not.
        self._scaler: torch.amp.GradScaler | None = (
            torch.amp.GradScaler(device) if precision == "fp16" else None
        )

    @property
    def enabled(self) -> bool:
        return self._dtype is not None

    def autocast(self) -> contextlib.AbstractContextManager:
        if self._dtype is None:
            return contextlib.nullcontext()
        # `torch.amp.autocast` ignores `device_type="cpu"` if dtype is fp16;
        # CPU autocast only supports bf16. Caller's responsibility.
        return torch.amp.autocast(self.device, dtype=self._dtype)

    def backward(self, loss: torch.Tensor) -> None:
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

    def unscale_(self, opt: torch.optim.Optimizer) -> None:
        """Call before grad-clip when using fp16."""
        if self._scaler is not None:
            self._scaler.unscale_(opt)

    def step(self, opt: torch.optim.Optimizer) -> None:
        if self._scaler is not None:
            self._scaler.step(opt)
        else:
            opt.step()

    def update(self) -> None:
        if self._scaler is not None:
            self._scaler.update()

    def clip_grad_norm(self, params, max_norm: float) -> None:
        """Wrapper that unscales first when fp16."""
        if self._scaler is not None:
            self._scaler.unscale_(_any_optim_from(params))
        nn.utils.clip_grad_norm_(params, max_norm)


def _any_optim_from(params):
    """`unscale_` needs the optimizer, not raw params. Algos should call
    `precision.unscale_(opt)` explicitly before clip; keep this for
    convenience-paths that pass params iterators."""
    raise NotImplementedError("Pass the optimizer to `unscale_` directly when using fp16.")


__all__ = ["Precision", "PrecisionType"]
