"""Check flexible-job-shop feasibility with the registered CP-SAT solver.

After arithmetic filtering, request a schedule with `feasibility_only=True`.
A valid reported cost indicates that the solver found a schedule. A timeout
without an incumbent does not establish infeasibility. If no solver is
registered, this helper returns the arithmetic result alone.
"""

from __future__ import annotations

from math import isnan
from typing import Any

import torch


def fjsp_cp_is_feasible(td: Any, *, time_limit_s: float = 3.0) -> torch.Tensor:
    """Return per-instance feasibility flags after arithmetic filtering.

    The registered solver searches for one feasible schedule within the time limit.
    Without a registered solver, the flags reflect arithmetic checks only.
    """
    from neuro_co.problems import BASELINE_SOLVERS, load_plugins

    load_plugins()  # ensure problem plug-ins register their solvers
    key = ("fjsp", "cpsat")
    from neuro_co.cax.feasibility.fjsp import fjsp_is_feasible

    arithmetic_ok = fjsp_is_feasible(td)
    if key not in BASELINE_SOLVERS or not arithmetic_ok.any():
        return arithmetic_ok

    solver = BASELINE_SOLVERS[key]
    B = int(td.batch_size[0])
    ok = arithmetic_ok.clone()
    for b in range(B):
        if not bool(arithmetic_ok[b]):
            continue
        instance = {
            k: td[k][b]
            for k in (
                "proc_times",
                "ops_ma_adj",
                "ops_job_map",
                "ops_sequence_order",
                "num_eligible",
                "pad_mask",
            )
            if k in td
        }
        try:
            _schedule, cost = solver(
                instance,
                max_runtime=time_limit_s,
                feasibility_only=True,
            )
            ok[b] = not (cost is None or (isinstance(cost, float) and isnan(cost)))
        except Exception:
            ok[b] = False
    return ok
