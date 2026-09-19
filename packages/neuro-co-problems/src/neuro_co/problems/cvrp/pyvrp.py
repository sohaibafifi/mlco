"""Sequential PyVRP adapter for depot-first CVRP corpora.

The public data contract deliberately uses NumPy-compatible arrays rather than
TensorDict objects:

* one instance has ``coords`` shaped ``[N + 1, 2]`` and ``demands`` shaped
  ``[N + 1]``;
* one corpus has ``coords`` shaped ``[B, N + 1, 2]`` and ``demands`` shaped
  ``[B, N + 1]``;
* location 0 is the depot and must have zero demand;
* routes contain customer indices only. Depot visits are implicit.

Both Euclidean distances and vehicle loads are rounded after multiplication by
``scaling_factor``. This preserves normalised demand/capacity inputs while
giving PyVRP the integer data it requires.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from pyvrp import Client, Depot, ProblemData, VehicleType
from pyvrp import solve as _solve
from pyvrp.stop import MaxIterations, MaxRuntime

DEFAULT_SCALING_FACTOR = 1_000_000
_MAX_SEED = 2**31 - 1


@dataclass(frozen=True, slots=True)
class CVRPSolveResult:
    """One feasible PyVRP result with depot-free routes.

    ``integer_cost`` is PyVRP's objective on the scaled distance matrix.
    ``cost`` is the corresponding distance in the input coordinate units.
    """

    instance_index: int
    routes: tuple[tuple[int, ...], ...]
    integer_cost: int
    cost: float
    seed: int
    limit_kind: str
    max_iterations: int | None
    max_runtime_s: float | None
    scaling_factor: int


def integer_distance_matrix(
    coords: Any,
    scaling_factor: int = DEFAULT_SCALING_FACTOR,
) -> np.ndarray:
    """Return the rounded, integer Euclidean matrix for one instance."""

    factor = _positive_integer("scaling_factor", scaling_factor)
    points = _instance_coords(coords)
    differences = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(differences, axis=-1)
    scaled = distances * factor
    _check_int64_range("scaled distances", scaled)
    matrix = np.rint(scaled).astype(np.int64)
    np.fill_diagonal(matrix, 0)
    return np.ascontiguousarray(matrix)


def build_problem_data(
    coords: Any,
    demands: Any,
    capacity: float,
    scaling_factor: int = DEFAULT_SCALING_FACTOR,
) -> ProblemData:
    """Build integer PyVRP data for one depot-first CVRP instance."""

    factor = _positive_integer("scaling_factor", scaling_factor)
    points = _instance_coords(coords)
    loads, vehicle_capacity = _scaled_loads(demands, capacity, points.shape[0], factor)

    scaled_coords = points * factor
    _check_int64_range("scaled coordinates", scaled_coords)
    integer_coords = np.rint(scaled_coords).astype(np.int64)

    depot = Depot(
        x=int(integer_coords[0, 0]),
        y=int(integer_coords[0, 1]),
        name="depot",
    )
    clients = [
        Client(
            x=int(integer_coords[index, 0]),
            y=int(integer_coords[index, 1]),
            delivery=[int(loads[index])],
            name=f"client_{index}",
        )
        for index in range(1, points.shape[0])
    ]
    vehicle_type = VehicleType(
        num_available=points.shape[0] - 1,
        capacity=[vehicle_capacity],
    )
    matrix = integer_distance_matrix(points, factor)
    return ProblemData(
        clients,
        [depot],
        [vehicle_type],
        [cast(Any, matrix)],
        [cast(Any, matrix)],
    )


def solution_routes(solution: Any) -> tuple[tuple[int, ...], ...]:
    """Convert a PyVRP solution to routes with no depot markers."""

    routes: list[tuple[int, ...]] = []
    for route in solution.routes():
        visits = tuple(int(customer) for customer in route.visits())
        if any(customer == 0 for customer in visits):
            raise ValueError("PyVRP route unexpectedly contains depot index 0")
        routes.append(visits)
    return tuple(routes)


def solve_corpus_sequential(
    coords: Any,
    demands: Any,
    capacity: float,
    *,
    seed: int,
    max_iterations: int | None = None,
    max_runtime_s: float | None = None,
    scaling_factor: int = DEFAULT_SCALING_FACTOR,
    collect_stats: bool = False,
) -> list[CVRPSolveResult]:
    """Solve a CVRP corpus sequentially under exactly one explicit limit.

    Instance ``i`` is solved with ``seed + i``. No worker pool or implicit
    parallelism is introduced by this adapter. The caller therefore controls
    the measured block around this function. Exactly one of ``max_iterations``
    and ``max_runtime_s`` must be supplied. A time-limited search receives a
    fresh stop criterion for every instance, so each instance gets the full
    requested runtime budget.
    """

    factor = _positive_integer("scaling_factor", scaling_factor)
    base_seed = _nonnegative_integer("seed", seed)
    iterations, runtime, limit_kind = _limits(max_iterations, max_runtime_s)

    corpus_coords = np.asarray(coords, dtype=np.float64)
    corpus_demands = np.asarray(demands, dtype=np.float64)
    if corpus_coords.ndim != 3 or corpus_coords.shape[2] != 2:
        raise ValueError(
            f"corpus coords must have shape [batch, nodes, 2], got {corpus_coords.shape}"
        )
    if corpus_coords.shape[0] < 1:
        raise ValueError("corpus must contain at least one instance")
    expected_demands = corpus_coords.shape[:2]
    if corpus_demands.shape != expected_demands:
        raise ValueError(
            f"corpus demands must have shape {expected_demands}, got {corpus_demands.shape}"
        )
    if base_seed + corpus_coords.shape[0] - 1 > _MAX_SEED:
        raise ValueError(
            f"seed plus corpus index must not exceed {_MAX_SEED}, "
            f"got {base_seed + corpus_coords.shape[0] - 1}"
        )

    outputs: list[CVRPSolveResult] = []
    for instance_index, (instance_coords, instance_demands) in enumerate(
        zip(corpus_coords, corpus_demands, strict=True)
    ):
        instance_seed = base_seed + instance_index
        data = build_problem_data(
            instance_coords,
            instance_demands,
            capacity,
            scaling_factor=factor,
        )
        if runtime is not None:
            stop = MaxRuntime(runtime)
        else:
            if iterations is None:  # pragma: no cover, guarded by _limits
                raise AssertionError("iteration limit unexpectedly absent")
            stop = MaxIterations(iterations)
        result = _solve(
            data,
            stop,
            seed=instance_seed,
            collect_stats=collect_stats,
            display=False,
        )
        if not result.is_feasible() or not result.best.is_complete():
            raise RuntimeError(
                f"PyVRP did not return a complete feasible solution for instance {instance_index}"
            )

        raw_cost = cast(Any, result.cost())
        try:
            integer_cost = operator.index(raw_cost)
        except TypeError as exc:
            raise TypeError(f"PyVRP returned a non-integer cost {raw_cost!r}") from exc
        outputs.append(
            CVRPSolveResult(
                instance_index=instance_index,
                routes=solution_routes(result.best),
                integer_cost=integer_cost,
                cost=integer_cost / factor,
                seed=instance_seed,
                limit_kind=limit_kind,
                max_iterations=iterations,
                max_runtime_s=runtime,
                scaling_factor=factor,
            )
        )

    return outputs


def _limits(
    max_iterations: int | None,
    max_runtime_s: float | None,
) -> tuple[int | None, float | None, str]:
    if (max_iterations is None) == (max_runtime_s is None):
        raise ValueError("provide exactly one of max_iterations or max_runtime_s")
    if max_runtime_s is not None:
        if isinstance(max_runtime_s, bool):
            raise TypeError(f"max_runtime_s must be a number, got {max_runtime_s!r}")
        try:
            runtime = float(max_runtime_s)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"max_runtime_s must be a number, got {max_runtime_s!r}") from exc
        if not math.isfinite(runtime) or runtime <= 0:
            raise ValueError(f"max_runtime_s must be finite and positive, got {max_runtime_s!r}")
        return None, runtime, "time"
    return _positive_integer("max_iterations", max_iterations), None, "iterations"


def _instance_coords(coords: Any) -> np.ndarray:
    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"instance coords must have shape [nodes, 2], got {points.shape}")
    if points.shape[0] < 2:
        raise ValueError("instance must contain a depot and at least one customer")
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
        raise ValueError(f"instance demands must have shape [{num_nodes}], got {loads.shape}")
    if not np.isfinite(loads).all():
        raise ValueError("demands must contain only finite values")
    if np.any(loads < 0):
        raise ValueError("demands must be nonnegative")
    if not np.isclose(loads[0], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"depot demand must be zero, got {loads[0]}")

    numeric_capacity = float(capacity)
    if not np.isfinite(numeric_capacity) or numeric_capacity <= 0:
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
        raise ValueError("scaling_factor is too small to represent the vehicle capacity")
    lost_positive = (loads[1:] > 0) & (scaled_loads[1:] == 0)
    if np.any(lost_positive):
        raise ValueError("scaling_factor is too small to represent a positive customer demand")
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
    "build_problem_data",
    "integer_distance_matrix",
    "solution_routes",
    "solve_corpus_sequential",
]
