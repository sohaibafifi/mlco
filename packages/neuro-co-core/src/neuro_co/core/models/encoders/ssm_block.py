"""Backend-agnostic Mamba block.

Selects the native `mamba_ssm` package when available (CUDA kernels, or
Apple MPS via the `mamba-ssm-macos` fork), else falls back to `mambapy`
(pure PyTorch, any device, slow). The ``"cuda"`` backend name is historic
and now means "native `mamba_ssm` kernel" regardless of device. Override
with env var ``NEURO_CO_MAMBA_BACKEND`` = ``cuda`` | ``mambapy``.
"""

from __future__ import annotations

import os
from typing import Literal

import torch
from torch import Tensor, nn

Backend = Literal["cuda", "mambapy"]


def get_backend(force: Backend | None = None) -> Backend:
    env = os.environ.get("NEURO_CO_MAMBA_BACKEND")
    if force is not None:
        choice = force
    elif env is not None:
        choice = env  # type: ignore[assignment]
    else:
        choice = "cuda" if _native_backend_available() else "mambapy"
    if choice not in ("cuda", "mambapy"):
        raise ValueError(f"unknown backend {choice!r}")
    return choice  # type: ignore[return-value]


def _native_backend_available() -> bool:
    """True if the native `mamba_ssm` package is importable and a GPU
    backend (CUDA, or Apple MPS via `mamba-ssm-macos`) is present."""
    has_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    if not has_gpu:
        return False
    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        return False
    return True


class SSMBlock(nn.Module):
    """Stacked Mamba S6 blocks. Input/output shape ``(B, L, D)``."""

    def __init__(
        self,
        d_model: int,
        n_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        backend: Backend | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.backend = get_backend(backend)
        self._impl = _build_impl(
            backend=self.backend,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (B,L,D), got shape {tuple(x.shape)}")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"last dim {x.shape[-1]} != d_model {self.d_model}")
        return self._impl(x)


def _build_impl(
    *,
    backend: Backend,
    d_model: int,
    n_layers: int,
    d_state: int,
    d_conv: int,
    expand: int,
) -> nn.Module:
    if backend == "cuda":
        return _build_cuda_stack(d_model, n_layers, d_state, d_conv, expand)
    return _build_mambapy_stack(d_model, n_layers, d_state, d_conv, expand)


def _build_cuda_stack(
    d_model: int, n_layers: int, d_state: int, d_conv: int, expand: int
) -> nn.Module:
    try:
        from mamba_ssm import Mamba
    except ImportError as exc:
        raise ImportError(
            "Native Mamba kernels require neuro-co-core[mamba-cuda] or neuro-co-core[mamba-macos]."
        ) from exc

    blocks = [
        Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        for _ in range(n_layers)
    ]
    return _Residual(nn.ModuleList(blocks), d_model)


def _build_mambapy_stack(
    d_model: int, n_layers: int, d_state: int, d_conv: int, expand: int
) -> nn.Module:
    try:
        from mambapy.mamba import Mamba, MambaConfig
    except ImportError as exc:
        raise ImportError("Install neuro-co-core[mamba] to use the Mamba encoder.") from exc

    cfg = MambaConfig(
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand_factor=expand,
    )
    return Mamba(cfg)


class _Residual(nn.Module):
    """Residual and LayerNorm wrapper for native `mamba_ssm.Mamba` blocks.

    `mambapy.mamba.Mamba` includes pre-normalization and residual connections.
    `mamba_ssm.Mamba` is a single block; we wrap it.
    """

    def __init__(self, blocks: nn.ModuleList, d_model: int):
        super().__init__()
        self.blocks = blocks
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(len(blocks))])

    def forward(self, x: Tensor) -> Tensor:
        for block, norm in zip(self.blocks, self.norms, strict=True):
            x = x + block(norm(x))
        return x
