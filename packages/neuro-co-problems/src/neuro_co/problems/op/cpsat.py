"""OR-Tools CP-SAT baseline for OP (orienteering problem).

Maximize collected prize subject to a tour-length budget. Modeled as
a binary visit-vector with TSP-like subtour elimination via MTZ
position variables. Small N (≤30 typical for OP benchmarks) makes
this tractable; for larger N consider a dedicated routing model.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any, Literal, overload

import numpy as np
from ortools.sat.python import cp_model

from neuro_co.core.result import SolveResult, ValidationReport
from neuro_co.problems.solver_result import cp_result


def _td_to_arrays(instance: Any) -> tuple[np.ndarray, np.ndarray, float]:
    locs = instance["locs"].cpu().numpy()  # [N, 2]
    prize = instance["prize"].cpu().numpy()  # [N], depot prize = 0
    max_len = float(instance["max_length"].cpu().numpy().max())
    return locs, prize, max_len


@overload
def solve_op(
    instance: Any,
    *,
    max_runtime: float = 5.0,
    scale: int = 10000,
    feasibility_only: bool = False,
    return_result: Literal[False] = False,
) -> tuple[list[int], float]: ...


@overload
def solve_op(
    instance: Any,
    *,
    max_runtime: float = 5.0,
    scale: int = 10000,
    feasibility_only: bool = False,
    return_result: Literal[True],
) -> SolveResult: ...


def solve_op(
    instance: Any,
    *,
    max_runtime: float = 5.0,
    scale: int = 10000,
    feasibility_only: bool = False,
    return_result: bool = False,
) -> tuple[list[int], float] | SolveResult:
    """Solve a single OP instance with CP-SAT.

    Parameters
    ----------
    feasibility_only
        If True, drop the prize-maximising objective and halt at the
        first feasible assignment. The solver returns a non-empty
        route iff a budget-respecting tour exists. Returned `cost` is
        `0.0` (no objective evaluated). Matches the VRPTW solver's
        `feasibility_only` flag and is used by
        `neuro_co.cax.feasibility.op_cp.op_cp_is_feasible` to turn a
        slow COP pass into a fast CSP feasibility-decision.

    Returns
    -------
    route
        Visited node indices in tour order (starts + ends at depot 0).
    cost
        `-prize_collected` (the reward convention: higher reward =
        more prize; we negate so the runner's `abs()` averaging works
        uniformly across solvers). When `feasibility_only=True`,
        returns `0.0` on success and `nan` on infeasibility.
    """
    locs, prize, max_len = _td_to_arrays(instance)
    n = locs.shape[0]
    if return_result and (
        locs.ndim != 2
        or locs.shape[1] != 2
        or n < 1
        or prize.shape != (n,)
        or not np.isfinite(locs).all()
        or not np.isfinite(prize).all()
        or np.any(prize < 0)
        or prize[0] != 0
        or not np.isfinite(max_len)
        or max_len < 0
    ):
        raise ValueError("OP requires finite coordinates/prizes and a nonnegative travel budget")
    diff = locs[:, None, :] - locs[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    dist_int = np.round(dist * scale).astype(np.int64)
    budget_int = round(max_len * scale)
    prize_int = np.round(prize * scale).astype(np.int64)

    model = cp_model.CpModel()
    # x[i, j] = 1 iff arc i->j used. Asymmetric (directed): OP is
    # routinely modeled as a directed tour to make MTZ clean.
    x = {(i, j): model.NewBoolVar(f"x_{i}_{j}") for i in range(n) for j in range(n) if i != j}
    visited = [model.NewBoolVar(f"v_{i}") for i in range(n)]

    # Depot always visited (start + end).
    if not return_result or feasibility_only:
        model.Add(visited[0] == 1)
    if feasibility_only and n > 1:
        # Exclude the vacuous depot-only circuit. The feasibility oracle
        # certifies that at least one prize-bearing customer can be visited.
        model.Add(sum(visited[1:]) >= 1)

    # Flow conservation: visited iff exactly one in-arc + one out-arc.
    for i in range(n):
        model.Add(sum(x[i, j] for j in range(n) if j != i) == visited[i])
        model.Add(sum(x[j, i] for j in range(n) if j != i) == visited[i])

    # MTZ subtour elimination (skip depot).
    u = [model.NewIntVar(0, n - 1, f"u_{i}") for i in range(n)]
    model.Add(u[0] == 0)
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            # u[i] - u[j] + n * x[i,j] <= n - 1
            model.Add(u[i] - u[j] + n * x[i, j] <= n - 1)

    # Tour-length budget.
    model.Add(
        sum(int(dist_int[i, j]) * x[i, j] for i in range(n) for j in range(n) if i != j)
        <= budget_int
    )

    # Objective: maximize total collected prize. Skipped under
    # `feasibility_only=True` so the solver stops at the first
    # budget-respecting tour rather than proving optimality (orders of
    # magnitude faster; matches the VRPTW `feasibility_only` path).
    if not feasibility_only:
        model.Maximize(sum(int(prize_int[i]) * visited[i] for i in range(n)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(max_runtime)
    solver.parameters.num_search_workers = 1
    if feasibility_only:
        solver.parameters.stop_after_first_solution = True
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if return_result:
            return cp_result(
                "op",
                solver,
                status,
                scale=scale,
                sense="max",
                metadata={
                    "model_matches_source": bool(
                        np.array_equal(dist_int / scale, dist)
                        and np.array_equal(prize_int / scale, prize)
                        and budget_int / scale == max_len
                    )
                },
            )
        return [], float("nan")
    if feasibility_only and not return_result:
        return [0], 0.0

    # Reconstruct tour: follow x[0, *].
    route = [0]
    visited_set = {0}
    cur = 0
    while True:
        nxt = next((j for j in range(n) if j != cur and solver.Value(x[cur, j])), None)
        if nxt is None or nxt == 0:
            route.append(0)
            break
        if nxt in visited_set:
            break
        route.append(nxt)
        visited_set.add(nxt)
        cur = nxt

    prize_collected = solver.ObjectiveValue() / scale
    if return_result:
        violations = []
        if (
            not route
            or route[0] != 0
            or route[-1] != 0
            or len(set(route[1:-1])) != len(route[1:-1])
        ):
            violations.append("Route must start and end at the depot without repeated customers")
        length = sum(float(dist[a, b]) for a, b in pairwise(route))
        if length > max_len + 1e-7:
            violations.append("Route exceeds the source travel budget")
        prize_value = float(sum(prize[i] for i in route if i != 0))
        validation = ValidationReport(
            feasible=not violations,
            objective=prize_value if not violations else None,
            violations=tuple(violations),
            metadata={"independent": True, "tour_length": length},
        )
        return cp_result(
            "op",
            solver,
            status,
            solution={"route": route},
            objective=prize_value,
            scale=scale,
            sense="max",
            validation=validation,
            feasibility_only=feasibility_only,
            metadata={
                "scale": scale,
                "model_matches_source": bool(
                    np.array_equal(dist_int / scale, dist)
                    and np.array_equal(prize_int / scale, prize)
                    and budget_int / scale == max_len
                ),
            },
        )
    return route, -float(prize_collected)
