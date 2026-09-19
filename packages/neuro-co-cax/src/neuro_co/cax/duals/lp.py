"""Generic LP-relaxation dual extractor (OR-Tools GLOP backend).

Sibling to `subgrad.py`. Per-problem files (`vrptw_lp.py`,
`fjsp_lp.py`, `op_lp.py`) hand a model-builder callback to
`solve_lp_and_extract_duals`; the core handles solver lifecycle,
status check, family-keyed dual aggregation, and the `mean |dual|`
reduction. Each per-problem file stays thin: it only declares
variables + constraints, never touches solver plumbing.

The callback signature is::

    builder(solver) -> dict[str, list[Constraint]]

mapping each constraint family name (`'capacity'`,
`'time_window'`, ...) to the list of solver constraint objects
that participate in it. The builder is also responsible for
setting the objective; the core then calls `solver.Solve()` and
walks the returned family table.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class LpResult:
    """Outcome of an LP-relaxation dual extraction."""

    multipliers: dict[str, float]  # family -> mean |dual|
    status: str  # 'optimal' | 'feasible' | 'infeasible' | 'unbounded' | 'error'
    objective_value: float | None
    wall_time_ms: int | None


def solve_lp_and_extract_duals(
    builder: Callable[[Any], dict[str, list[Any]]],
    *,
    time_limit_s: float = 5.0,
    family_default: dict[str, float] | None = None,
    aggregate: str = "mean",
) -> LpResult:
    """Build + solve an LP, return family-keyed `mean |dual|` table.

    Parameters
    ----------
    builder
        Callable taking the freshly-created `pywraplp.Solver` and
        populating it with variables, the objective, and a
        `{family: [Constraint, ...]}` mapping (returned).
    time_limit_s
        Solver time budget in seconds.
    family_default
        Default value per family when the LP is infeasible /
        unbounded / errors out. Defaults to 0.0 per family.

    Returns
    -------
    LpResult
    """
    try:
        from ortools.linear_solver import pywraplp
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "solve_lp_and_extract_duals needs the `ortools` extra. "
            "Install via `uv sync --all-extras`."
        ) from exc

    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        raise RuntimeError("OR-Tools GLOP solver unavailable")
    solver.SetTimeLimit(int(time_limit_s * 1000))

    family_rows = builder(solver)
    if not isinstance(family_rows, dict):
        raise TypeError(
            f"builder must return dict[family_name, [Constraint]]; got {type(family_rows).__name__}"
        )

    status_code = solver.Solve()
    status_map = {
        pywraplp.Solver.OPTIMAL: "optimal",
        pywraplp.Solver.FEASIBLE: "feasible",
        pywraplp.Solver.INFEASIBLE: "infeasible",
        pywraplp.Solver.UNBOUNDED: "unbounded",
        pywraplp.Solver.ABNORMAL: "abnormal",
        pywraplp.Solver.NOT_SOLVED: "not_solved",
    }
    status = status_map.get(status_code, "error")

    if status not in ("optimal", "feasible"):
        defaults = family_default or {k: 0.0 for k in family_rows}
        return LpResult(
            multipliers={k: float(defaults.get(k, 0.0)) for k in family_rows},
            status=status,
            objective_value=None,
            wall_time_ms=int(solver.wall_time()),
        )

    mults = {family: _agg_abs_dual(rows, aggregate) for family, rows in family_rows.items()}
    return LpResult(
        multipliers=mults,
        status=status,
        objective_value=float(solver.Objective().Value()),
        wall_time_ms=int(solver.wall_time()),
    )


def mean_abs_dual(rows: list[Any]) -> float:
    """Mean of `|constraint.dual_value()|` across `rows`, ignoring non-finite."""
    vals = [abs(c.dual_value()) for c in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _agg_abs_dual(rows: list[Any], how: str) -> float:
    """Family-row dual aggregation. `how` in {'mean', 'sum', 'max'}."""
    vals = [abs(c.dual_value()) for c in rows]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return 0.0
    if how == "mean":
        return float(sum(vals) / len(vals))
    if how == "sum":
        return float(sum(vals))
    if how == "max":
        return float(max(vals))
    raise ValueError(f"unknown aggregate={how!r}; expected 'mean', 'sum', or 'max'")
