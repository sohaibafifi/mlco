"""OR-Tools CP routing baseline for PDP (pickup-delivery).

Uses `pywrapcp.RoutingModel.AddPickupAndDelivery` with a single
vehicle visiting all pairs (the `PDPEnv` is a single-tour
formulation). Capacity isn't part of basic PDP; only the
pickup-before-delivery precedence constraint matters.
"""

from __future__ import annotations

import math
import time
from itertools import pairwise
from typing import Any, Literal, overload

import numpy as np

from neuro_co.core.result import SolveResult, SolveStatus, ValidationReport


def _td_to_arrays(instance: Any) -> tuple[np.ndarray, int]:
    """Extract `[N, 2]` locations and infer the number of pairs P."""
    locs = instance["locs"].cpu().numpy()
    n = int(locs.shape[0])
    p = (n - 1) // 2
    return locs, p


@overload
def solve_pdp(
    instance: Any,
    *,
    max_runtime: float = 5.0,
    scale: int = 10000,
    return_result: Literal[False] = False,
) -> tuple[list[int], float]: ...


@overload
def solve_pdp(
    instance: Any,
    *,
    max_runtime: float = 5.0,
    scale: int = 10000,
    return_result: Literal[True],
) -> SolveResult: ...


def solve_pdp(
    instance: Any,
    *,
    max_runtime: float = 5.0,
    scale: int = 10000,
    return_result: bool = False,
) -> tuple[list[int], float] | SolveResult:
    """Solve a single PDP instance with OR-Tools routing.

    Returns
    -------
    route
        Node indices in visit order, starting + ending at depot.
    cost
        `-tour_length` (the reward convention).
    """
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    started = time.perf_counter()
    locs, p = _td_to_arrays(instance)
    n = locs.shape[0]
    if return_result and (
        locs.ndim != 2 or locs.shape[1] != 2 or n < 3 or n % 2 != 1 or not np.isfinite(locs).all()
    ):
        raise ValueError("PDP requires finite [1 + 2 * pairs, 2] coordinates and at least one pair")

    # Integer distance matrix.
    diff = locs[:, None, :] - locs[None, :, :]
    dist = np.linalg.norm(diff, axis=-1) * scale
    dist_int = np.round(dist).astype(np.int64)

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def _dist_cb(from_index: int, to_index: int) -> int:
        return int(dist_int[manager.IndexToNode(from_index), manager.IndexToNode(to_index)])

    transit_cb_idx = routing.RegisterTransitCallback(_dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    # Pickup-delivery precedence: visit i then i+P, same vehicle.
    order_cb_idx = routing.RegisterTransitCallback(lambda _origin, _destination: 1)
    routing.AddDimension(order_cb_idx, 0, n, True, "VisitOrder")
    distance_dim = routing.GetDimensionOrDie("VisitOrder")
    for i in range(1, p + 1):
        pickup_idx = manager.NodeToIndex(i)
        delivery_idx = manager.NodeToIndex(i + p)
        routing.AddPickupAndDelivery(pickup_idx, delivery_idx)
        routing.solver().Add(routing.VehicleVar(pickup_idx) == routing.VehicleVar(delivery_idx))
        routing.solver().Add(
            distance_dim.CumulVar(pickup_idx) <= distance_dim.CumulVar(delivery_idx)
        )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    if return_result:
        if not math.isfinite(max_runtime) or max_runtime <= 0:
            raise ValueError("max_runtime must be finite and positive")
        nanoseconds = math.ceil(max_runtime * 1_000_000_000)
        params.time_limit.seconds = nanoseconds // 1_000_000_000
        params.time_limit.nanos = nanoseconds % 1_000_000_000
    else:
        params.time_limit.seconds = max(1, int(max_runtime))

    solution = routing.SolveWithParameters(params)
    if solution is None:
        if return_result:
            status = int(routing.status())
            return SolveResult(
                problem="pdp",
                method="ortools",
                status=SolveStatus.INFEASIBLE
                if status == 6
                else SolveStatus.INVALID
                if status == 5
                else SolveStatus.UNKNOWN,
                feasible=False if status == 6 else None,
                solver_status=f"ROUTING_STATUS_{status}",
                runtime_s=time.perf_counter() - started,
            )
        return [], float("nan")

    route: list[int] = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        route.append(manager.IndexToNode(idx))
        idx = solution.Value(routing.NextVar(idx))
    route.append(manager.IndexToNode(idx))  # depot close
    cost = solution.ObjectiveValue() / scale
    if return_result:
        violations = []
        if sorted(route[1:-1]) != list(range(1, n)) or route[0] != 0 or route[-1] != 0:
            violations.append("Route does not cover every customer exactly once")
        if not violations and any(route.index(i) > route.index(i + p) for i in range(1, p + 1)):
            violations.append("A delivery precedes its pickup")
        objective = sum(float(np.linalg.norm(locs[a] - locs[b])) for a, b in pairwise(route))
        validation = ValidationReport(
            feasible=not violations,
            objective=objective if not violations else None,
            violations=tuple(violations),
            metadata={"independent": True},
        )
        optimal = int(routing.status()) == 7 and np.array_equal(dist_int / scale, dist / scale)
        return SolveResult(
            problem="pdp",
            method="ortools",
            status=SolveStatus.INVALID
            if violations
            else SolveStatus.OPTIMAL
            if optimal
            else SolveStatus.FEASIBLE,
            solution={"route": route},
            objective=validation.objective,
            feasible=validation.feasible,
            validation=validation,
            bound=objective if optimal else None,
            gap=0.0 if optimal else None,
            solver_status=f"ROUTING_STATUS_{int(routing.status())}",
            runtime_s=time.perf_counter() - started,
            metadata={"scale": scale},
        )
    return route, -float(cost)
