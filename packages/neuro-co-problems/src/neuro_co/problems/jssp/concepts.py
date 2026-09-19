"""JSSP concept extractors.

Each returns `[B, num_ops]` long labels in `{0, 1, -1}` (`-1` = padded
op, ignored by the probe trainer). `None` is returned when the
required field is missing.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _mark_padded(labels: Tensor, pad_mask: Tensor) -> Tensor:
    out = labels.clone()
    out[pad_mask] = -1
    return out


def _binary_above_median_masked(values: Tensor, pad_mask: Tensor) -> Tensor:
    valid = values[~pad_mask]
    if valid.numel() == 0:
        return torch.full_like(values, -1, dtype=torch.long)
    threshold = valid.median()
    labels = (values > threshold).long()
    return _mark_padded(labels, pad_mask)


def _binary_below_median_masked(values: Tensor, pad_mask: Tensor) -> Tensor:
    valid = values[~pad_mask]
    if valid.numel() == 0:
        return torch.full_like(values, -1, dtype=torch.long)
    threshold = valid.median()
    labels = (values < threshold).long()
    return _mark_padded(labels, pad_mask)


def _op_proc_time(td: Any) -> Tensor | None:
    """Per-op processing time on its assigned machine. `[B, num_ops]`."""
    if "proc_times" not in td or "ops_ma_adj" not in td:
        return None
    return (td["proc_times"] * td["ops_ma_adj"]).sum(dim=-2)


def long_proc_time(td: Any) -> Tensor | None:
    """Op's processing time above the median valid op duration."""
    duration = _op_proc_time(td)
    if duration is None or duration.ndim != 2:
        return None
    pad = td.get("pad_mask", torch.zeros_like(duration, dtype=torch.bool))
    return _binary_above_median_masked(duration, pad)


def high_machine_load(td: Any) -> Tensor | None:
    """Op assigned to a machine whose total load is above median.

    Machine load = sum of `proc_times` for all ops scheduled on it.
    Each op inherits its machine's load label.
    """
    if "proc_times" not in td or "ops_ma_adj" not in td:
        return None
    machine_load = (td["proc_times"] * td["ops_ma_adj"]).sum(dim=-1)  # [B, M]
    op_load = (td["ops_ma_adj"] * machine_load.unsqueeze(-1)).sum(dim=-2)  # [B, N]
    if op_load.ndim != 2:
        return None
    pad = td.get("pad_mask", torch.zeros_like(op_load, dtype=torch.bool))
    return _binary_above_median_masked(op_load, pad)


def low_slack(td: Any) -> Tensor | None:
    """Op with low lower-bound completion (`lbs`); close to the critical path."""
    if "lbs" not in td:
        return None
    lbs = td["lbs"]
    if lbs.ndim != 2:
        return None
    pad = td.get("pad_mask", torch.zeros_like(lbs, dtype=torch.bool))
    return _binary_below_median_masked(lbs, pad)


CONCEPTS = {
    "long_proc_time": long_proc_time,
    "high_machine_load": high_machine_load,
    "low_slack": low_slack,
}
