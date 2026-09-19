"""Check CVRPTW feasibility with a time-limited CP-SAT search.

An arithmetic filter runs first. The solver then searches for one feasible
route set without an optimization objective. A returned route set certifies
feasibility; timeout or failure to find one does not prove infeasibility.
When the solver import is unavailable, this helper returns the arithmetic
result alone.
"""

from __future__ import annotations

from typing import Any

import torch


def vrptw_cp_is_feasible(td: Any, *, time_limit_s: float = 1.0) -> torch.Tensor:
    """Per-batch CSP feasibility-decision check. Returns `[B]` bool tensor.

    Falls back to the arithmetic check
    (`neuro_co.cax.feasibility.vrptw.vrptw_is_feasible`) when the
    `ortools` extra is not installed; the pipeline still runs, just
    without the constructive feasibility certificate.
    """
    from neuro_co.cax.feasibility.vrptw import vrptw_is_feasible

    # Step 1: cheap arithmetic pre-filter. Any element failing here
    # is infeasible at the structural level; no need to invoke CSP.
    arithmetic_ok = vrptw_is_feasible(td)
    if not arithmetic_ok.any():
        return arithmetic_ok

    # Search for one feasible route set per surviving instance.
    try:
        from neuro_co.problems.vrptw.cpsat import solve_cvrptw_cpsat
    except ImportError:  # pragma: no cover - ortools is an optional extra
        return arithmetic_ok

    B = int(td.batch_size[0])
    cp_ok = arithmetic_ok.clone()
    for b in range(B):
        if not bool(arithmetic_ok[b]):
            continue
        try:
            single = _slice_batch(td, b)
            routes, _cost = solve_cvrptw_cpsat(
                single,
                max_runtime=float(time_limit_s),
                feasibility_only=True,
            )
            cp_ok[b] = bool(routes)  # non-empty route list = feasible
        except Exception:
            cp_ok[b] = False
    return cp_ok


def _slice_batch(td: Any, idx: int) -> Any:
    """Return a one-instance slice while preserving batch metadata."""
    return td[idx : idx + 1]
