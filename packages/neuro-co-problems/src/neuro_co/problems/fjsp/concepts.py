"""FJSP (flexible job-shop) concept extractors (core `FJSPState`).

`state.proc_times[B, O, M]` carries processing time per (op, machine), zero
on ineligible pairs; `state.ops_ma_adj[B, O, M]` is the eligibility mask.
Concepts reduce over the machine axis (-1). Labels `[B, O]` in `{0, 1, -1}`,
or `None` when a field is absent. (Core FJSP has no padding, so all real ops.)
"""

from __future__ import annotations

from typing import Any

from torch import Tensor


def _above_median(values: Tensor) -> Tensor:
    return (values > values.median()).long()


def _eligible_proc_stats(state: Any) -> tuple[Tensor, Tensor] | None:
    """(mean_proc, var_proc) per op over eligible machines."""
    proc = getattr(state, "proc_times", None)
    if proc is None or proc.ndim != 3:
        return None
    eligible = proc > 0  # [B, O, M]
    counts = eligible.sum(dim=-1).clamp(min=1).float()  # [B, O]
    mean_p = (proc * eligible).sum(dim=-1) / counts
    var_p = (((proc - mean_p.unsqueeze(-1)) ** 2) * eligible).sum(dim=-1) / counts
    return mean_p, var_p


def long_proc_time(state: Any) -> Tensor | None:
    stats = _eligible_proc_stats(state)
    if stats is None:
        return None
    return _above_median(stats[0])


def high_flexibility(state: Any) -> Tensor | None:
    """Number of eligible machines per op above the median."""
    n_elig = getattr(state, "num_eligible", None)
    if n_elig is None:
        proc = getattr(state, "proc_times", None)
        if proc is None:
            return None
        n_elig = (proc > 0).sum(dim=-1)
    return _above_median(n_elig.float())


def high_proc_variance(state: Any) -> Tensor | None:
    """Per-op processing-time variance across eligible machines above median."""
    stats = _eligible_proc_stats(state)
    if stats is None:
        return None
    return _above_median(stats[1])


CONCEPTS = {
    "long_proc_time": long_proc_time,
    "high_flexibility": high_flexibility,
    "high_proc_variance": high_proc_variance,
}
