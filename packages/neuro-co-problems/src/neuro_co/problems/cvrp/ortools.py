"""Independent OR-Tools Routing adapter for depot-first CVRP corpora.

The array and result contracts mirror :mod:`neuro_co.problems.cvrp.pyvrp`:

* coordinates have shape ``[B, N + 1, 2]`` and demands ``[B, N + 1]``;
* node 0 is the depot and routes contain customer indices only;
* Euclidean distances and loads are rounded after multiplication by one
  explicit scaling factor;
* corpus instances are solved sequentially with seed ``base_seed + index``.

OR-Tools is an optional dependency and is imported only when solving. A
solution-count limit gives a reproducible search cutoff for a fixed OR-Tools
version and seed. A time limit is also supported and recorded, but its final
incumbent can depend on machine timing.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

DEFAULT_SCALING_FACTOR = 1_000_000
_MAX_SEED = 2**31 - 1
_SEARCH_STATUS_NAMES = {
    0: "ROUTING_NOT_SOLVED",
    1: "ROUTING_SUCCESS",
    2: "ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED",
    3: "ROUTING_FAIL",
    4: "ROUTING_FAIL_TIMEOUT",
    5: "ROUTING_INVALID",
    6: "ROUTING_INFEASIBLE",
    7: "ROUTING_OPTIMAL",
}


@dataclass(frozen=True, slots=True)
class CVRPSolveResult:
    """One feasible OR-Tools result with depot-free routes and explicit limits."""

    instance_index: int
    routes: tuple[tuple[int, ...], ...]
    integer_cost: int
    cost: float
    seed: int
    scaling_factor: int
    limit_kind: str
    max_runtime_s: float | None
    solution_limit: int | None
    search_status: str


def integer_distance_matrix(
    coords: Any,
    scaling_factor: int = DEFAULT_SCALING_FACTOR,
) -> np.ndarray:
    """Return rounded integer Euclidean distances for one instance."""

    factor = _positive_integer("scaling_factor", scaling_factor)
    points = _instance_coords(coords)
    differences = points[:, None, :] - points[None, :, :]
    scaled = np.linalg.norm(differences, axis=-1) * factor
    _check_int64_range("scaled distances", scaled)
    matrix = np.rint(scaled).astype(np.int64)
    np.fill_diagonal(matrix, 0)
    return np.ascontiguousarray(matrix)


def solve_corpus_sequential(
    coords: Any,
    demands: Any,
    capacity: float,
    *,
    seed: int,
    max_runtime_s: float | None = None,
    solution_limit: int | None = None,
    scaling_factor: int = DEFAULT_SCALING_FACTOR,
) -> list[CVRPSolveResult]:
    """Solve a CVRP corpus sequentially under exactly one explicit limit.

    ``solution_limit`` is the preferred reproducible pilot limit. Exactly one
    of ``solution_limit`` and ``max_runtime_s`` must be supplied. The latter is
    useful for time-budget policies but is inherently sensitive to wall-clock
    scheduling even when the search seed is fixed.
    """

    factor = _positive_integer("scaling_factor", scaling_factor)
    base_seed = _nonnegative_integer("seed", seed)
    runtime, solutions, limit_kind = _limits(max_runtime_s, solution_limit)

    corpus_coords = np.asarray(coords, dtype=np.float64)
    corpus_demands = np.asarray(demands, dtype=np.float64)
    if corpus_coords.ndim != 3 or corpus_coords.shape[2] != 2:
        raise ValueError(
            f"corpus coords must have shape [batch, nodes, 2], got {corpus_coords.shape}"
        )
    if corpus_coords.shape[0] < 1:
        raise ValueError("corpus must contain at least one instance")
    if corpus_demands.shape != corpus_coords.shape[:2]:
        raise ValueError(
            f"corpus demands must have shape {corpus_coords.shape[:2]}, got {corpus_demands.shape}"
        )
    if base_seed + corpus_coords.shape[0] - 1 > _MAX_SEED:
        raise ValueError(
            f"seed plus corpus index must not exceed {_MAX_SEED}, "
            f"got {base_seed + corpus_coords.shape[0] - 1}"
        )

    results: list[CVRPSolveResult] = []
    for instance_index, (instance_coords, instance_demands) in enumerate(
        zip(corpus_coords, corpus_demands, strict=True)
    ):
        results.append(
            _solve_instance(
                instance_coords,
                instance_demands,
                capacity,
                instance_index=instance_index,
                seed=base_seed + instance_index,
                max_runtime_s=runtime,
                solution_limit=solutions,
                scaling_factor=factor,
                limit_kind=limit_kind,
            )
        )
    return results


def _solve_instance(
    coords: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    *,
    instance_index: int,
    seed: int,
    max_runtime_s: float | None,
    solution_limit: int | None,
    scaling_factor: int,
    limit_kind: str,
) -> CVRPSolveResult:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    matrix = integer_distance_matrix(coords, scaling_factor)
    loads, vehicle_capacity = _scaled_loads(
        demands,
        capacity,
        matrix.shape[0],
        scaling_factor,
    )
    num_vehicles = matrix.shape[0] - 1
    manager = pywrapcp.RoutingIndexManager(matrix.shape[0], num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        origin = manager.IndexToNode(from_index)
        destination = manager.IndexToNode(to_index)
        return int(matrix[origin, destination])

    transit_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    def demand_callback(from_index: int) -> int:
        return int(loads[manager.IndexToNode(from_index)])

    demand_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_index,
        0,
        [vehicle_capacity] * num_vehicles,
        True,
        "Capacity",
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.log_search = False
    params.sat_parameters.random_seed = seed
    params.sat_parameters.num_search_workers = 1
    if max_runtime_s is not None:
        total_nanoseconds = math.ceil(max_runtime_s * 1_000_000_000)
        params.time_limit.seconds = total_nanoseconds // 1_000_000_000
        params.time_limit.nanos = total_nanoseconds % 1_000_000_000
    else:
        if solution_limit is None:  # pragma: no cover, guarded by _limits
            raise AssertionError("solution limit unexpectedly absent")
        params.solution_limit = solution_limit

    routing.solver().ReSeed(seed)
    assignment = routing.SolveWithParameters(params)
    status_code = int(routing.status())
    status = _SEARCH_STATUS_NAMES.get(status_code, f"ROUTING_STATUS_{status_code}")
    if assignment is None:
        raise RuntimeError(
            f"OR-Tools returned no feasible CVRP solution for instance {instance_index}: {status}"
        )

    routes = _solution_routes(manager, routing, assignment, num_vehicles)
    _validate_routes(routes, loads, vehicle_capacity, matrix.shape[0] - 1)
    integer_cost = _route_cost(routes, matrix)
    objective = operator.index(assignment.ObjectiveValue())
    if objective != integer_cost:
        raise RuntimeError(
            f"OR-Tools objective disagrees with extracted routes for instance {instance_index}"
        )
    return CVRPSolveResult(
        instance_index=instance_index,
        routes=routes,
        integer_cost=integer_cost,
        cost=integer_cost / scaling_factor,
        seed=seed,
        scaling_factor=scaling_factor,
        limit_kind=limit_kind,
        max_runtime_s=max_runtime_s,
        solution_limit=solution_limit,
        search_status=status,
    )


def _solution_routes(
    manager: Any,
    routing: Any,
    assignment: Any,
    num_vehicles: int,
) -> tuple[tuple[int, ...], ...]:
    routes: list[tuple[int, ...]] = []
    for vehicle in range(num_vehicles):
        index = routing.Start(vehicle)
        route: list[int] = []
        while not routing.IsEnd(index):
            node = int(manager.IndexToNode(index))
            if node != 0:
                route.append(node)
            index = assignment.Value(routing.NextVar(index))
        if route:
            routes.append(tuple(route))
    return tuple(routes)


def _route_cost(routes: tuple[tuple[int, ...], ...], matrix: np.ndarray) -> int:
    total = 0
    for route in routes:
        sequence = (0, *route, 0)
        total += sum(int(matrix[origin, destination]) for origin, destination in pairwise(sequence))
    return total


def _validate_routes(
    routes: tuple[tuple[int, ...], ...],
    loads: np.ndarray,
    capacity: int,
    num_clients: int,
) -> None:
    customers = [customer for route in routes for customer in route]
    if sorted(customers) != list(range(1, num_clients + 1)):
        raise RuntimeError("OR-Tools routes do not cover every customer exactly once")
    if any(sum(int(loads[customer]) for customer in route) > capacity for route in routes):
        raise RuntimeError("OR-Tools returned a route above vehicle capacity")


def _limits(
    max_runtime_s: float | None,
    solution_limit: int | None,
) -> tuple[float | None, int | None, str]:
    if (max_runtime_s is None) == (solution_limit is None):
        raise ValueError("provide exactly one of max_runtime_s or solution_limit")
    if max_runtime_s is not None:
        if isinstance(max_runtime_s, bool):
            raise TypeError(f"max_runtime_s must be a number, got {max_runtime_s!r}")
        runtime = float(max_runtime_s)
        if not math.isfinite(runtime) or runtime <= 0:
            raise ValueError(f"max_runtime_s must be finite and positive, got {max_runtime_s!r}")
        return runtime, None, "time"
    return None, _positive_integer("solution_limit", solution_limit), "solutions"


def _instance_coords(coords: Any) -> np.ndarray:
    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"coords must have shape [nodes, 2], got {points.shape}")
    if points.shape[0] < 2:
        raise ValueError("coords must contain a depot and at least one customer")
    if not np.isfinite(points).all():
        raise ValueError("coords must contain only finite values")
    return points


def _scaled_loads(
    demands: Any,
    capacity: float,
    num_nodes: int,
    scaling_factor: int,
) -> tuple[np.ndarray, int]:
    loads = np.asarray(demands, dtype=np.float64)
    if loads.shape != (num_nodes,):
        raise ValueError(f"demands must have shape [{num_nodes}], got {loads.shape}")
    if not np.isfinite(loads).all():
        raise ValueError("demands must contain only finite values")
    if np.any(loads < 0):
        raise ValueError("demands must be nonnegative")
    if not np.isclose(loads[0], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"depot demand must be zero, got {loads[0]}")

    numeric_capacity = float(capacity)
    if not math.isfinite(numeric_capacity) or numeric_capacity <= 0:
        raise ValueError(f"capacity must be finite and positive, got {capacity!r}")
    if np.any(loads[1:] > numeric_capacity):
        raise ValueError("each customer demand must not exceed vehicle capacity")

    scaled_loads_float = loads * scaling_factor
    scaled_capacity_float = numeric_capacity * scaling_factor
    _check_int64_range("scaled demands", scaled_loads_float)
    _check_int64_range("scaled capacity", np.asarray([scaled_capacity_float]))
    scaled_loads = np.rint(scaled_loads_float).astype(np.int64)
    scaled_capacity = round(scaled_capacity_float)
    if scaled_capacity <= 0:
        raise ValueError("scaling_factor is too small to represent vehicle capacity")
    if np.any((loads[1:] > 0) & (scaled_loads[1:] == 0)):
        raise ValueError("scaling_factor is too small to represent a positive demand")
    if np.any(scaled_loads[1:] > scaled_capacity):
        raise ValueError("rounded customer demand exceeds rounded vehicle capacity")
    return scaled_loads, scaled_capacity


def _check_int64_range(name: str, values: np.ndarray) -> None:
    limit = np.iinfo(np.int64).max
    if not np.isfinite(values).all() or np.any(np.abs(values) > limit):
        raise ValueError(f"{name} cannot be represented as signed 64-bit integers")


def _positive_integer(name: str, value: Any) -> int:
    parsed = _nonnegative_integer(name, value)
    if parsed == 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return parsed


def _nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative, got {parsed}")
    return parsed


__all__ = [
    "DEFAULT_SCALING_FACTOR",
    "CVRPSolveResult",
    "integer_distance_matrix",
    "solve_corpus_sequential",
]
