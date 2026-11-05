"""Independent route checks using input distances, capacities, and windows."""

from __future__ import annotations

from collections import Counter
from itertools import pairwise
from numbers import Integral
from typing import Any

import numpy as np

from neuro_co.core.result import ValidationReport

from .instances import RoutingInstance


def validate_routes(instance: RoutingInstance, routes: Any) -> ValidationReport:
    """Check complete depot-free routes and recompute their total objective.

    A TSP solution has one route containing every node except fixed origin zero.
    CVRP/CVRPTW allow multiple nonempty routes. Time windows constrain service
    start and depot return; each vehicle departs at the depot's earliest time
    after its service duration. Tolerance is relative to numeric input scale.
    """
    violations: list[str] = []
    if not isinstance(routes, (tuple, list, np.ndarray)):
        return ValidationReport(False, None, ("routes must be a sequence of routes",))
    if isinstance(routes, np.ndarray) and routes.ndim == 0:
        return ValidationReport(False, None, ("routes must be a sequence of routes",))
    if instance.problem == "tsp" and len(routes) != 1:
        violations.append("TSP requires exactly one route")
    if len(routes) == 0:
        violations.append("solution has no routes")
    parsed: list[tuple[int, ...]] = []
    nodes: list[int] = []
    for index, route in enumerate(routes):
        if not isinstance(route, (tuple, list, np.ndarray)) or (
            isinstance(route, np.ndarray) and route.ndim != 1
        ):
            violations.append(f"route {index} must be a one-dimensional sequence")
            continue
        if not len(route):
            violations.append(f"route {index} is empty")
            continue
        if any(
            isinstance(node, (bool, np.bool_)) or not isinstance(node, Integral) for node in route
        ):
            violations.append(f"route {index} contains a noninteger node")
            continue
        values = tuple(int(node) for node in route)
        if any(node < 1 or node >= instance.num_nodes for node in values):
            violations.append(f"route {index} contains depot zero or an out-of-range node")
            continue
        parsed.append(values)
        nodes.extend(values)
    counts = Counter(nodes)
    missing = sorted(set(range(1, instance.num_nodes)) - counts.keys())
    duplicates = sorted(node for node, count in counts.items() if count > 1)
    if missing:
        violations.append(f"missing nodes: {missing}")
    if duplicates:
        violations.append(f"duplicate nodes: {duplicates}")
    distances = instance.distance_matrix()
    capacity_tolerance = 1e-9 * max(1.0, float(instance.capacity or 0))
    time_tolerance = 1e-9 * max(
        1.0,
        float(distances.max()),
        float(instance.time_windows.max()) if instance.time_windows is not None else 0,
    )
    objective = 0.0
    loads: list[float] = []
    for index, route in enumerate(parsed):
        sequence = (0, *route, 0)
        objective += sum(float(distances[a, b]) for a, b in pairwise(sequence))
        if instance.demand is not None:
            load = float(instance.demand[list(route)].sum())
            loads.append(load)
            assert instance.capacity is not None
            if load > instance.capacity + capacity_tolerance:
                violations.append(f"route {index} load {load} exceeds capacity {instance.capacity}")
        if instance.time_windows is not None:
            windows = instance.time_windows
            assert instance.service_time is not None
            time = float(windows[0, 0] + instance.service_time[0])
            for previous, node in pairwise(sequence):
                arrival = time + float(distances[previous, node])
                service_start = max(arrival, float(windows[node, 0]))
                if service_start > windows[node, 1] + time_tolerance:
                    violations.append(f"route {index} misses time window at node {node}")
                time = service_start + (float(instance.service_time[node]) if node != 0 else 0.0)
    return ValidationReport(
        feasible=not violations,
        objective=objective if not violations else None,
        violations=tuple(violations),
        metadata={
            "distance_convention": instance.distance_convention,
            "route_loads": loads,
            "num_routes": len(routes),
            "num_nodes": instance.num_nodes,
        },
    )


__all__ = ["validate_routes"]
