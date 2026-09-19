"""Route recomposition and lightweight repair."""

from __future__ import annotations

import numpy as np

from .types import Partition, Solution, VRPInstance


def nearest_neighbor_route(instance: VRPInstance, customers: tuple[int, ...]) -> tuple[int, ...]:
    """Order a customer set with a deterministic nearest-neighbor heuristic."""

    remaining = set(customers)
    route: list[int] = []
    current = 0
    while remaining:
        nxt = min(remaining, key=lambda c: (edge_cost(instance, current, c), c))
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return tuple(route)


def compose_nearest_neighbor(
    instance: VRPInstance,
    partition: Partition,
    *,
    repair_capacity: bool = True,
) -> Solution:
    """Build one or more global routes per cluster."""

    routes: list[tuple[int, ...]] = []
    for cluster in partition.clusters:
        route = nearest_neighbor_route(instance, cluster)
        if repair_capacity:
            routes.extend(split_over_capacity(instance, route))
        else:
            routes.append(route)
    return Solution(
        routes=tuple(r for r in routes if r),
        metadata={"partition_method": partition.method, "repair_capacity": repair_capacity},
    )


def time_window_aware_routes(
    instance: VRPInstance,
    customers: tuple[int, ...],
    *,
    repair_capacity: bool = True,
) -> tuple[tuple[int, ...], ...]:
    """Build routes with a greedy time-window-aware ordering."""

    remaining = set(customers)
    routes: list[tuple[int, ...]] = []
    while remaining:
        route: list[int] = []
        current = 0
        load = 0.0
        time = 0.0
        while remaining:
            feasible = [
                _candidate_score(instance, current, time, customer)
                for customer in remaining
                if _candidate_is_feasible(
                    instance,
                    current,
                    time,
                    customer,
                    load=load,
                    repair_capacity=repair_capacity,
                )
            ]
            if not feasible:
                if route:
                    break
                forced = min(
                    remaining,
                    key=lambda customer: _candidate_score(instance, current, time, customer),
                )
                route.append(forced)
                time = _departure_time(instance, current, time, forced)
                remaining.remove(forced)
                break
            _, _, _, _, _, customer = min(feasible)
            route.append(customer)
            time = _departure_time(instance, current, time, customer)
            load += float(instance.demand[customer])
            current = customer
            remaining.remove(customer)
        if route:
            routes.append(tuple(route))
    return tuple(routes)


def compose_time_window_aware(
    instance: VRPInstance,
    partition: Partition,
    *,
    repair_capacity: bool = True,
) -> Solution:
    """Build global routes with a time-window-aware local constructor."""

    routes: list[tuple[int, ...]] = []
    for cluster in partition.clusters:
        routes.extend(time_window_aware_routes(instance, cluster, repair_capacity=repair_capacity))
    return Solution(
        routes=tuple(r for r in routes if r),
        metadata={
            "partition_method": partition.method,
            "repair_capacity": repair_capacity,
            "constructor": "time_window",
        },
    )


def split_over_capacity(
    instance: VRPInstance, route: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    """Split a route whenever adding the next customer would exceed capacity."""

    routes: list[tuple[int, ...]] = []
    current: list[int] = []
    load = 0.0
    for customer in route:
        demand = float(instance.demand[customer])
        if current and load + demand > instance.capacity:
            routes.append(tuple(current))
            current = []
            load = 0.0
        current.append(customer)
        load += demand
    if current:
        routes.append(tuple(current))
    return tuple(routes)


def actions_from_solution(solution: Solution) -> list[int]:
    """Flatten depot-free routes into a depot-separated action sequence."""

    actions: list[int] = []
    for route in solution.routes:
        actions.append(0)
        actions.extend(route)
    actions.append(0)
    return actions


def edge_cost(instance: VRPInstance, i: int, j: int) -> float:
    if instance.cost_matrix is not None:
        return float(instance.cost_matrix[i, j])
    diff = instance.coords[i] - instance.coords[j]
    distance = float(np.sqrt(np.dot(diff, diff)))
    # TSPLIB EUC_2D rounds to the nearest integer; with integer coordinates the
    # exact .5 tie cannot occur, so plain rounding matches nint().
    return round(distance) if instance.round_distances else distance


def _candidate_is_feasible(
    instance: VRPInstance,
    current: int,
    time: float,
    customer: int,
    *,
    load: float,
    repair_capacity: bool,
) -> bool:
    if repair_capacity and load + float(instance.demand[customer]) > instance.capacity + 1e-9:
        return False
    service_start = _service_start(instance, current, time, customer)
    return not (
        instance.tw_late is not None and service_start > float(instance.tw_late[customer]) + 1e-9
    )


def _candidate_score(
    instance: VRPInstance,
    current: int,
    time: float,
    customer: int,
) -> tuple[float, float, float, float, float, int]:
    arrival = time + edge_cost(instance, current, customer)
    early = 0.0 if instance.tw_early is None else float(instance.tw_early[customer])
    late = float("inf") if instance.tw_late is None else float(instance.tw_late[customer])
    service_start = max(arrival, early)
    lateness = 0.0 if instance.tw_late is None else max(service_start - late, 0.0)
    wait = max(early - arrival, 0.0)
    return (lateness, late, service_start, wait, edge_cost(instance, current, customer), customer)


def _departure_time(instance: VRPInstance, current: int, time: float, customer: int) -> float:
    service_start = _service_start(instance, current, time, customer)
    service_time = 0.0 if instance.service_time is None else float(instance.service_time[customer])
    return service_start + service_time


def _service_start(instance: VRPInstance, current: int, time: float, customer: int) -> float:
    arrival = time + edge_cost(instance, current, customer)
    if instance.tw_early is None:
        return arrival
    return max(arrival, float(instance.tw_early[customer]))
