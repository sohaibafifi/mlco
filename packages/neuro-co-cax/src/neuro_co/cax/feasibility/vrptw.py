"""Necessary arithmetic checks for batched CVRPTW instances.

Check nonnegative demands and service times, ordered time-window endpoints,
positive capacity, and aggregate demand against a loose fleet-capacity bound.
Only fields present in the instance mapping are checked. Passing these tests
does not establish that a feasible route exists.
"""

from __future__ import annotations

from typing import Any

import torch


def vrptw_is_feasible(td: Any) -> torch.Tensor:
    """Element-wise arithmetic check on a batched CVRPTW instance dict."""
    B = int(td["locs"].shape[0]) if "locs" in td else int(td["demand"].shape[0])
    ok = torch.ones(B, dtype=torch.bool)

    if "demand" in td:
        demand = td["demand"]
        ok = ok & (demand >= 0).all(dim=-1).cpu()
    if "time_windows" in td:
        tw = td["time_windows"]
        # tw shape [B, N, 2]: open <= close
        ok = ok & (tw[..., 0] <= tw[..., 1]).all(dim=-1).cpu()
    if "durations" in td:
        dur = td["durations"]
        ok = ok & (dur >= 0).all(dim=-1).cpu()
    if "vehicle_capacity" in td:
        cap = td["vehicle_capacity"]
        # Squeeze any trailing singleton dim.
        if cap.ndim > 1:
            cap = cap.squeeze(-1)
        ok = ok & (cap > 0).cpu()
        # Total demand <= K * Q (need enough fleet capacity overall).
        if "demand" in td:
            total = td["demand"].sum(dim=-1).cpu()
            # Use one available vehicle per customer as a loose fleet bound.
            n_customers = int(td["demand"].shape[-1])
            ok = ok & (total <= float(n_customers) * cap.cpu())
    return ok
