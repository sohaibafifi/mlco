"""Core `State` -> canonical instance-dict adapter.

The CP feasibility checks and LP / subgradient dual backends operate on a
plain instance dict (numpy / tensor fields), not on a policy state. This
module localises the mapping from a core `State` (+ its `Env`, which holds
problem constants like vehicle capacity and travel budget) to that dict, so
the solver code stays problem-math only.

Canonical keys per problem (match what the dual / feasibility backends read):

- cvrptw/vrptw: locs, demand, time_windows[...,2], durations, vehicle_capacity
- op:           locs, prize, max_length
- fjsp:         proc_times, num_eligible, ops_ma_adj, pad_mask,
                ops_job_map, ops_sequence_order
"""

from __future__ import annotations

from typing import Any

import torch


def to_batch_instance(state: Any, env: Any, problem: str) -> dict[str, torch.Tensor]:
    """Batched instance dict `[B, ...]` from a core `State` + `Env`."""
    key = problem.lower()
    if key in ("cvrptw", "vrptw"):
        cap = float(getattr(env, "capacity", 1.0))
        b = state.coords.shape[0]
        return {
            "locs": state.coords,
            "demand": state.demand,
            "time_windows": torch.stack([state.tw_early, state.tw_late], dim=-1),
            "durations": torch.zeros_like(state.demand),
            "vehicle_capacity": torch.full((b,), cap, device=state.coords.device),
        }
    if key == "op":
        budget = float(getattr(env, "budget", 1.0))
        b = state.coords.shape[0]
        return {
            "locs": state.coords,
            "prize": state.prize,
            "max_length": torch.full((b,), budget, device=state.coords.device),
        }
    if key == "fjsp":
        return {
            "proc_times": state.proc_times,
            "num_eligible": state.num_eligible,
            "ops_ma_adj": state.ops_ma_adj,
            "pad_mask": torch.zeros_like(state.op_done),  # no padding in core FJSP
            "ops_job_map": state.job_id,
            "ops_sequence_order": state.op_in_job,
        }
    raise KeyError(f"no instance adapter for problem={problem!r}")


def to_instance(state: Any, env: Any, problem: str, batch_idx: int = 0) -> dict[str, Any]:
    """Single-instance numpy dict (the `batch_idx`-th), for per-instance solvers."""
    batched = to_batch_instance(state, env, problem)
    out: dict[str, Any] = {}
    for k, v in batched.items():
        arr = v.detach().cpu().numpy()
        out[k] = arr[batch_idx] if arr.ndim > 0 and arr.shape[0] > batch_idx else arr
    return out


__all__ = ["to_batch_instance", "to_instance"]
