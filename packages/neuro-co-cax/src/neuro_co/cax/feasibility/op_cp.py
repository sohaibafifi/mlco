"""Check orienteering feasibility with the registered CP-SAT solver.

After arithmetic filtering, request a feasible tour with `feasibility_only=True`.
The helper accepts a nonempty route with a valid reported cost and rejects
solver failures or timeouts. If no solver is registered, it returns the
arithmetic result alone.
"""

from __future__ import annotations

from math import isnan
from typing import Any

import torch


def op_cp_is_feasible(td: Any, *, time_limit_s: float = 2.0) -> torch.Tensor:
    """Return per-instance feasibility flags after arithmetic filtering.

    The registered solver searches for one feasible tour within the time limit.
    Without a registered solver, the flags reflect arithmetic checks only.
    """
    from neuro_co.problems import BASELINE_SOLVERS, load_plugins

    load_plugins()  # ensure problem plug-ins register their solvers
    key = ("op", "cpsat")
    from neuro_co.cax.feasibility.op import op_is_feasible

    # Stage 1: cheap arithmetic pre-filter. Any element failing here
    # is infeasible at the structural level; no need to invoke CSP.
    arithmetic_ok = op_is_feasible(td)
    if key not in BASELINE_SOLVERS or not arithmetic_ok.any():
        return arithmetic_ok

    solver = BASELINE_SOLVERS[key]
    B = int(td.batch_size[0])
    ok = arithmetic_ok.clone()
    for b in range(B):
        if not bool(arithmetic_ok[b]):
            continue
        instance = {
            "locs": td["locs"][b],
            "prize": td["prize"][b],
            "max_length": td["max_length"][b] if "max_length" in td else torch.tensor(2.0),
        }
        try:
            route, cost = solver(
                instance,
                max_runtime=time_limit_s,
                feasibility_only=True,
            )
            # `feasibility_only` returns a non-empty route + cost=0.0
            # on feasibility, empty + nan on infeasibility / timeout.
            ok[b] = bool(route) and not (cost is None or (isinstance(cost, float) and isnan(cost)))
        except Exception:
            ok[b] = False
    return ok
