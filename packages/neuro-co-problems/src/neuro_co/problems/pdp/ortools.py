"""OR-Tools CP routing baseline for PDP (pickup-delivery).

Uses `pywrapcp.RoutingModel.AddPickupAndDelivery` with a single
vehicle visiting all pairs (the `PDPEnv` is a single-tour
formulation). Capacity isn't part of basic PDP; only the
pickup-before-delivery precedence constraint matters.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _td_to_arrays(instance: Any) -> tuple[np.ndarray, int]:
    """Extract `[N, 2]` locations and infer the number of pairs P."""
    locs = instance["locs"].cpu().numpy()
    n = int(locs.shape[0])
    p = (n - 1) // 2
    return locs, p


def solve_pdp(
    instance: Any,
    *,
    max_runtime: float = 5.0,
    scale: int = 10000,
) -> tuple[list[int], float]:
    """Solve a single PDP instance with OR-Tools routing.

    Returns
    -------
    route
        Node indices in visit order, starting + ending at depot.
    cost
        `-tour_length` (the reward convention).
    """
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    locs, p = _td_to_arrays(instance)
    n = locs.shape[0]

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
    routing.AddDimension(transit_cb_idx, 0, 10**9, True, "Distance")
    distance_dim = routing.GetDimensionOrDie("Distance")
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
    params.time_limit.seconds = max(1, int(max_runtime))

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return [], float("nan")

    route: list[int] = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        route.append(manager.IndexToNode(idx))
        idx = solution.Value(routing.NextVar(idx))
    route.append(manager.IndexToNode(idx))  # depot close
    cost = solution.ObjectiveValue() / scale
    return route, -float(cost)
