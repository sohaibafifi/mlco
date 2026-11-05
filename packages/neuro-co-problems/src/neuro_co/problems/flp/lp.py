"""OR-Tools CBC/MIP solver for the FLP (k-median) env.

ILP formulation:

    min   sum_{i,j} d_ij * x_ij
    s.t.  sum_j  x_ij = 1         forall i           (customer i assigned)
          sum_i  y_i  = k                            (k facilities open)
          x_ij <= y_j              forall i, j        (assigned only if open)
          x_ij, y_i in {0,1}

We use `pywraplp` with the CBC backend. CBC's branch-and-cut on
this structure typically finds the optimum in under 1 s on
N=100, k=10 instances. The earlier CP-SAT attempt timed out
because the 10,000 binary `x_ij` vars exceeded its
preprocessing budget; CBC's stronger LP relaxation + cut
generators handle k-median directly.

A linear-program-relaxation fallback (`solver=GLOP`) is
available via `relax_only=True` -- useful for *lower-bound*
checks but not as a baseline (returns fractional facilities).
"""

from __future__ import annotations

import math
import time
from typing import Any, Literal, overload

import numpy as np

from neuro_co.core.result import SolveResult, SolveStatus, ValidationReport

_DIST_SCALE = 1000


@overload
def solve_flp(
    instance: Any,
    *,
    max_runtime: float = 5.0,
    relax_only: bool = False,
    return_result: Literal[False] = False,
) -> tuple[list[int], float]: ...


@overload
def solve_flp(
    instance: Any,
    *,
    max_runtime: float = 5.0,
    relax_only: bool = False,
    return_result: Literal[True],
) -> SolveResult: ...


def solve_flp(
    instance: Any,
    *,
    max_runtime: float = 5.0,
    relax_only: bool = False,
    return_result: bool = False,
) -> tuple[list[int], float] | SolveResult:
    """Solve a single FLP instance with OR-Tools CBC (or GLOP for LP-relax).

    Returns `(chosen_facilities, -total_min_dist)` matching the
    the reward sign (negative cost). On failure / time-out
    without an incumbent: `([], nan)`.
    """
    try:
        from ortools.linear_solver import pywraplp
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError("solve_flp needs OR-Tools (`uv sync --all-extras`).") from exc

    started = time.perf_counter()
    locs = instance["locs"].cpu().numpy()
    if locs.ndim == 3 and locs.shape[0] == 1:
        locs = locs[0]
    k_arr = instance["to_choose"]
    k = int(k_arr.cpu().item()) if hasattr(k_arr, "cpu") else int(k_arr)
    n = locs.shape[0]
    if k <= 0 or k >= n:
        if return_result:
            return SolveResult(
                problem="flp",
                method="lp",
                status=SolveStatus.INVALID,
                solver_status="INVALID_FACILITY_COUNT",
            )
        return list(range(min(k, n))), float("nan")

    diff = locs[:, None, :] - locs[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))  # [N, N] float

    backend = "GLOP" if relax_only else "CBC"
    solver = pywraplp.Solver.CreateSolver(backend)
    if solver is None:
        if return_result:
            return SolveResult(
                problem="flp",
                method="lp",
                status=SolveStatus.ERROR,
                solver_status="BACKEND_UNAVAILABLE",
            )
        return [], float("nan")
    solver.SetTimeLimit(int(max_runtime * 1000))

    # ---- Variables ----
    if relax_only:
        y = [solver.NumVar(0.0, 1.0, f"y_{i}") for i in range(n)]
        x = [[solver.NumVar(0.0, 1.0, f"x_{i}_{j}") for j in range(n)] for i in range(n)]
    else:
        y = [solver.IntVar(0, 1, f"y_{i}") for i in range(n)]
        x = [[solver.IntVar(0, 1, f"x_{i}_{j}") for j in range(n)] for i in range(n)]

    # ---- Constraints ----
    # Each customer assigned to exactly one facility.
    for i in range(n):
        solver.Add(solver.Sum(x[i][j] for j in range(n)) == 1)
    # Exactly k facilities open.
    solver.Add(solver.Sum(y) == k)
    # x_ij <= y_j: customer i can only be assigned to an open facility j.
    for i in range(n):
        for j in range(n):
            solver.Add(x[i][j] <= y[j])

    # ---- Objective ----
    solver.Minimize(solver.Sum(float(dist[i, j]) * x[i][j] for i in range(n) for j in range(n)))

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        if return_result:
            mapped = {
                pywraplp.Solver.INFEASIBLE: SolveStatus.INFEASIBLE,
                pywraplp.Solver.MODEL_INVALID: SolveStatus.INVALID,
            }.get(status, SolveStatus.UNKNOWN)
            return SolveResult(
                problem="flp",
                method="lp",
                status=mapped,
                feasible=False if mapped == SolveStatus.INFEASIBLE else None,
                solver_status=f"MPSOLVER_STATUS_{status}",
                runtime_s=time.perf_counter() - started,
            )
        return [], float("nan")

    chosen = [i for i in range(n) if y[i].solution_value() > 0.5]
    cost = -float(solver.Objective().Value())  # negative -> the sign
    if return_result:
        if relax_only:
            return SolveResult(
                problem="flp",
                method="lp",
                status=SolveStatus.UNKNOWN,
                bound=-cost if status == pywraplp.Solver.OPTIMAL else None,
                solver_status=f"GLOP_STATUS_{status}",
                runtime_s=time.perf_counter() - started,
                metadata={"relaxation": True, "relaxation_objective": -cost},
            )
        valid = len(chosen) == k and len(set(chosen)) == k
        objective = float(dist[:, chosen].min(axis=1).sum()) if valid else None
        validation = ValidationReport(
            feasible=valid,
            objective=objective,
            violations=() if valid else ("Incorrect number of open facilities",),
            metadata={"independent": True},
        )
        bound = float(solver.Objective().BestBound())
        if not math.isfinite(bound):
            bound = None
        return SolveResult(
            problem="flp",
            method="lp",
            status=SolveStatus.INVALID
            if not valid
            else SolveStatus.OPTIMAL
            if status == pywraplp.Solver.OPTIMAL
            else SolveStatus.FEASIBLE,
            solution={"facilities": chosen},
            objective=objective,
            feasible=valid,
            validation=validation,
            bound=bound,
            gap=abs(objective - bound) / max(abs(objective), 1e-12)
            if objective is not None and bound is not None
            else None,
            solver_status=f"CBC_STATUS_{status}",
            runtime_s=time.perf_counter() - started,
        )
    _ = _DIST_SCALE  # legacy, kept for backward compat with CP-SAT branch
    return chosen, cost
