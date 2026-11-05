"""Structured results shared by optional classical solver adapters."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import numpy as np

from neuro_co.core.result import SolveResult, SolveStatus, ValidationReport


def cp_result(
    problem: str,
    solver: Any,
    status: Any,
    *,
    solution: dict[str, Any] | None = None,
    objective: float | None = None,
    scale: float = 1.0,
    sense: str = "min",
    feasibility_only: bool = False,
    validation: ValidationReport | None = None,
    metadata: dict[str, Any] | None = None,
) -> SolveResult:
    """Preserve CP-SAT's termination reason and objective units."""
    name = solver.StatusName(status)
    mapped = {
        "OPTIMAL": SolveStatus.OPTIMAL,
        "FEASIBLE": SolveStatus.FEASIBLE,
        "INFEASIBLE": SolveStatus.INFEASIBLE,
        "MODEL_INVALID": SolveStatus.INVALID,
    }.get(name, SolveStatus.UNKNOWN)
    feasible = True if name in {"OPTIMAL", "FEASIBLE"} else False if name == "INFEASIBLE" else None
    model_matches = (metadata or {}).get("model_matches_source", True)
    if not model_matches and name == "INFEASIBLE":
        mapped, feasible = SolveStatus.UNKNOWN, None
    if not model_matches and feasible:
        mapped = SolveStatus.FEASIBLE
    bound = None
    if feasible and not feasibility_only and model_matches:
        candidate = float(solver.BestObjectiveBound()) / scale
        if math.isfinite(candidate):
            bound = candidate
    if feasibility_only and feasible:
        mapped = SolveStatus.FEASIBLE
    has_incumbent = feasible is True
    if validation is not None and has_incumbent:
        feasible = validation.feasible
        objective = validation.objective
        if not feasible:
            mapped, bound = SolveStatus.INVALID, None
    gap = None
    if objective is not None and bound is not None:
        gap = abs(objective - bound) / max(abs(objective), 1e-12)
    return SolveResult(
        problem=problem,
        method="cpsat",
        status=mapped,
        solution=solution if has_incumbent else None,
        objective=objective if feasible else None,
        sense=sense,
        bound=bound,
        gap=gap,
        runtime_s=float(solver.WallTime()),
        feasible=feasible,
        validation=validation,
        solver_status=name,
        metadata={"feasibility_only": feasibility_only, **(metadata or {})},
    )


def validate_schedule(
    schedule: list[tuple[int, int, float]],
    processing: np.ndarray,
    job_map: np.ndarray,
    order: np.ndarray,
    active: list[int],
) -> ValidationReport:
    """Check operation coverage, eligibility, precedences and machine overlaps."""
    violations: list[str] = []
    if sorted(item[0] for item in schedule) != sorted(active):
        violations.append("Schedule must contain every active operation exactly once")
    values: dict[int, tuple[int, float, float]] = {}
    for op, machine, start in schedule:
        if not (0 <= op < processing.shape[1] and 0 <= machine < processing.shape[0]):
            violations.append("Schedule contains an invalid operation or machine")
            continue
        duration = float(processing[machine, op])
        if duration <= 0 or not math.isfinite(duration):
            violations.append(f"Operation {op} uses an ineligible machine")
        if not math.isfinite(start) or start < 0:
            violations.append(f"Operation {op} has an invalid start time")
        values[op] = (machine, float(start), float(start) + duration)
    jobs: dict[int, list[int]] = {}
    machines: dict[int, list[tuple[float, float, int]]] = {}
    for op, (machine, start, end) in values.items():
        jobs.setdefault(int(job_map[op]), []).append(op)
        machines.setdefault(machine, []).append((start, end, op))
    for ops in jobs.values():
        ops.sort(key=lambda op: int(order[op]))
        for a, b in pairwise(ops):
            if values[a][2] > values[b][1] + 1e-7:
                violations.append(f"Job precedence violated between operations {a} and {b}")
    for intervals in machines.values():
        intervals.sort()
        for a, b in pairwise(intervals):
            if a[1] > b[0] + 1e-7:
                violations.append(f"Machine overlap between operations {a[2]} and {b[2]}")
    objective = max((item[2] for item in values.values()), default=0.0)
    return ValidationReport(
        feasible=not violations,
        objective=objective if not violations else None,
        violations=tuple(violations),
        metadata={"independent": True},
    )


def source_matrix(coords: Any, cost_matrix: Any | None) -> np.ndarray:
    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("coords must have shape [nodes, 2] with at least two nodes")
    if not np.isfinite(points).all():
        raise ValueError("coords must be finite")
    matrix = (
        np.linalg.norm(points[:, None] - points[None, :], axis=-1)
        if cost_matrix is None
        else np.asarray(cost_matrix, dtype=np.float64)
    )
    if matrix.shape != (len(points), len(points)):
        raise ValueError("cost_matrix must be square with one row per node")
    if not np.isfinite(matrix).all() or np.any(matrix < 0) or np.any(np.diag(matrix) != 0):
        raise ValueError("cost_matrix must be finite, nonnegative, and zero on the diagonal")
    return matrix


def source_loads(demands: Any, num_nodes: int, capacity: float) -> np.ndarray:
    """Validate source CVRP data before applying any necessary-condition proof."""
    loads = np.asarray(demands, dtype=np.float64)
    if loads.shape != (num_nodes,) or not np.isfinite(loads).all() or np.any(loads < 0):
        raise ValueError("demands must be a finite nonnegative vector with one value per node")
    if not np.isclose(loads[0], 0.0, atol=1e-12, rtol=0):
        raise ValueError("depot demand must be zero")
    if isinstance(capacity, bool) or not math.isfinite(capacity) or capacity <= 0:
        raise ValueError("capacity must be finite and positive")
    return loads


def validate_cvrp(
    routes: tuple[tuple[int, ...], ...], matrix: np.ndarray, demands: Any, capacity: float
) -> ValidationReport:
    """Recompute feasibility and objective in the caller's original units."""
    loads = np.asarray(demands, dtype=np.float64)
    violations = []
    customers = [node for route in routes for node in route]
    if sorted(customers) != list(range(1, len(matrix))):
        violations.append("Routes must visit every customer exactly once")
    valid_indices = all(0 < node < len(matrix) for node in customers)
    if not valid_indices:
        violations.append("Route contains an invalid customer index")
    if valid_indices:
        for index, route in enumerate(routes):
            if sum(float(loads[node]) for node in route) > capacity + 1e-9:
                violations.append(f"Route {index} exceeds vehicle capacity")
    objective = None
    if valid_indices and not violations:
        objective = sum(
            float(matrix[a, b])
            for route in routes
            for a, b in zip((0, *route), (*route, 0), strict=True)
        )
    return ValidationReport(
        feasible=not violations,
        objective=objective,
        violations=tuple(violations),
        metadata={"independent": True},
    )


__all__ = ["cp_result", "source_loads", "source_matrix", "validate_cvrp", "validate_schedule"]
