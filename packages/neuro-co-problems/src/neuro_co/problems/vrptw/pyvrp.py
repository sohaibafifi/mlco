"""PyVRP baseline solver for the core CVRPTW instance dict.

Consumes the canonical instance dict produced by
`neuro_co.cax._instance.to_instance(state, env, "cvrptw")`:

  - `locs`             [N, 2]    depot at index 0
  - `demand`           [N]       per node (depot = 0)
  - `time_windows`     [N, 2]    (early, late) per node
  - `durations`        [N]       service durations
  - `vehicle_capacity` scalar

Self-contained: no TensorDict. PyVRP is an optional (`cp`) extra.
"""

from __future__ import annotations

import time
from typing import Any, Literal, overload

import numpy as np
from pyvrp import Client, Depot, ProblemData, VehicleType
from pyvrp import solve as _solve
from pyvrp.stop import MaxRuntime

from neuro_co.core.result import SolveResult, SolveStatus
from neuro_co.problems.instances import RoutingInstance
from neuro_co.problems.solver_result import source_matrix
from neuro_co.problems.validation import validate_routes

# Integer scaling for PyVRP's integer solver (coords/time in [0, 1]-ish range).
SCALING_FACTOR = 1000


def _scale(x: Any, factor: int) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) * factor).round().astype(np.int64)


def _cost_matrix(coords_scaled: np.ndarray) -> np.ndarray:
    diff = coords_scaled[:, None, :] - coords_scaled[None, :, :]
    return np.sqrt((diff.astype(np.float64) ** 2).sum(-1)).round().astype(np.int64)


def instance2data(
    instance: dict[str, Any], scaling_factor: int, *, cost_matrix: Any | None = None
) -> ProblemData:
    """Build PyVRP `ProblemData` from a core CVRPTW instance dict."""
    coords = _scale(instance["locs"], scaling_factor)
    demand = _scale(instance["demand"], scaling_factor)
    tw = _scale(instance["time_windows"], scaling_factor)
    durations = _scale(instance["durations"], scaling_factor)
    capacity = int(_scale(instance["vehicle_capacity"], scaling_factor).reshape(-1)[0])
    num_locs = int(coords.shape[0])

    depot = Depot(x=int(coords[0][0]), y=int(coords[0][1]))
    clients = [
        Client(
            x=int(coords[idx][0]),
            y=int(coords[idx][1]),
            tw_early=int(tw[idx][0]),
            tw_late=int(tw[idx][1]),
            delivery=[int(demand[idx])],
            service_duration=int(durations[idx]),
            name=f"client_{idx + 1}",
        )
        for idx in range(1, num_locs)
    ]
    vehicle_type = VehicleType(
        num_available=num_locs - 1,
        capacity=[capacity],
        tw_early=int(tw[0][0]),
        tw_late=int(tw[0][1]),
    )
    matrix = (
        _cost_matrix(coords)
        if cost_matrix is None
        else _scale(source_matrix(instance["locs"], cost_matrix), scaling_factor)
    )
    return ProblemData(clients, [depot], [vehicle_type], [matrix], [matrix])


def _solution2action(solution: Any) -> list[int]:
    """Flatten PyVRP routes into a depot-separated visiting sequence."""
    action: list[int] = []
    for route in solution.routes():
        action.append(0)  # depot
        action.extend(int(v) for v in route.visits())
    action.append(0)
    return action


@overload
def solve_cvrptw(
    instance: dict[str, Any],
    max_runtime: float,
    *,
    return_result: Literal[False] = False,
    cost_matrix: Any | None = None,
) -> tuple[list[int], float]: ...


@overload
def solve_cvrptw(
    instance: dict[str, Any],
    max_runtime: float,
    *,
    return_result: Literal[True],
    cost_matrix: Any | None = None,
) -> SolveResult: ...


def solve_cvrptw(
    instance: dict[str, Any],
    max_runtime: float,
    *,
    return_result: bool = False,
    cost_matrix: Any | None = None,
) -> tuple[list[int], float] | SolveResult:
    started = time.perf_counter()
    data = instance2data(instance, SCALING_FACTOR, cost_matrix=cost_matrix)
    result = _solve(data, MaxRuntime(max_runtime), display=False)
    if return_result:
        if not result.is_feasible() or not result.best.is_complete():
            return SolveResult(
                problem="cvrptw",
                method="pyvrp",
                status=SolveStatus.UNKNOWN,
                feasible=None,
                solver_status="NO_FEASIBLE_INCUMBENT",
                runtime_s=time.perf_counter() - started,
            )
        matrix = source_matrix(instance["locs"], cost_matrix)
        routing = RoutingInstance(
            problem="cvrptw",
            coords=np.asarray(instance["locs"]),
            demand=np.asarray(instance["demand"]),
            capacity=float(np.asarray(instance["vehicle_capacity"]).reshape(-1)[0]),
            time_windows=np.asarray(instance["time_windows"]),
            service_time=np.asarray(instance["durations"]),
            cost_matrix=matrix,
            distance_convention="explicit",
        )
        routes = tuple(tuple(int(v) for v in route.visits()) for route in result.best.routes())
        validation = validate_routes(routing, routes)
        return SolveResult(
            problem="cvrptw",
            method="pyvrp",
            status=SolveStatus.FEASIBLE if validation.feasible else SolveStatus.INVALID,
            feasible=validation.feasible,
            validation=validation,
            solution={"routes": [list(route) for route in routes]},
            objective=validation.objective,
            solver_status="FEASIBLE_INCUMBENT",
            runtime_s=time.perf_counter() - started,
            metadata={"scaling_factor": SCALING_FACTOR, "integer_objective": int(result.cost())},
        )
    action = _solution2action(result.best)
    cost = -result.cost() / SCALING_FACTOR
    return action, cost
