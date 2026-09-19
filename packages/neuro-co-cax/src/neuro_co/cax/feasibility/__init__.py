"""Per-problem CO instance-feasibility checks for cp_counterfactual.

`is_feasible(state, env, problem)` returns a `[B]` bool tensor saying, per
batch element, whether the instance satisfies the problem's structural
constraints (non-negative demand, ordered time windows, positive capacity,
non-negative processing times, ...).

The check reads a canonical instance dict produced from the core `State` +
`Env` by `neuro_co.cax._instance.to_batch_instance`. Arithmetic mode is
cheap field-value sanity; `cp_sat` mode (OR-Tools, the `cp` extra) upgrades
to a decision query.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from math import isnan
from typing import Any

import torch

from neuro_co.cax._instance import to_batch_instance, to_instance


def is_feasible(
    state: Any,
    env: Any,
    problem: str,
    *,
    mode: str = "arithmetic",
    time_limit_s: float = 1.0,
    max_workers: int = 1,
) -> torch.Tensor:
    """Per-batch feasibility check. Returns `[B]` bool tensor."""
    key = problem.lower()
    if mode not in ("arithmetic", "cp_sat"):
        raise ValueError(f"mode must be 'arithmetic' or 'cp_sat'; got {mode!r}")

    if mode == "arithmetic":
        inst = to_batch_instance(state, env, problem)
        if key in ("vrptw", "cvrptw"):
            from neuro_co.cax.feasibility.vrptw import vrptw_is_feasible

            return vrptw_is_feasible(inst)
        if key == "op":
            from neuro_co.cax.feasibility.op import op_is_feasible

            return op_is_feasible(inst)
        if key == "fjsp":
            from neuro_co.cax.feasibility.fjsp import fjsp_is_feasible

            return fjsp_is_feasible(inst)
        b = state.coords.shape[0] if hasattr(state, "coords") else 1
        return torch.ones(b, dtype=torch.bool)

    # cp_sat: per-instance OR-Tools decision query (the `cp` extra).
    b = state.coords.shape[0] if hasattr(state, "coords") else state.proc_times.shape[0]
    instances = [to_instance(state, env, problem, batch_idx=i) for i in range(b)]
    worker_count = max(1, min(int(max_workers), b))
    if worker_count == 1:
        flags = [_cp_one(inst, key, time_limit_s) for inst in instances]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            flags = list(
                executor.map(
                    lambda inst: _cp_one(inst, key, time_limit_s),
                    instances,
                )
            )
    return torch.tensor(flags, dtype=torch.bool)


def _cp_one(inst: dict[str, Any], key: str, time_limit_s: float) -> bool:
    tensor_inst = {name: torch.as_tensor(value) for name, value in inst.items()}
    if key in ("vrptw", "cvrptw"):
        from neuro_co.problems.vrptw.cpsat import solve_cvrptw_cpsat

        routes, cost = solve_cvrptw_cpsat(
            tensor_inst,
            max_runtime=time_limit_s,
            feasibility_only=True,
        )
        return bool(routes) and not isnan(float(cost))
    if key == "op":
        from neuro_co.problems.op.cpsat import solve_op

        route, cost = solve_op(
            tensor_inst,
            max_runtime=time_limit_s,
            feasibility_only=True,
        )
        return bool(route) and not isnan(float(cost))
    if key == "fjsp":
        from neuro_co.problems.fjsp.cpsat import solve_fjsp

        _schedule, cost = solve_fjsp(
            tensor_inst,
            max_runtime=time_limit_s,
            feasibility_only=True,
        )
        return not isnan(float(cost))
    return True


__all__ = ["is_feasible"]
