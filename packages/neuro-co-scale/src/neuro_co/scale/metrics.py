"""Quality, feasibility, and scale metrics for VRP solutions."""

from __future__ import annotations

from collections import Counter
from itertools import pairwise

from .compose import edge_cost
from .types import ScaleMetrics, Solution, VRPInstance


def route_cost(instance: VRPInstance, route: tuple[int, ...]) -> float:
    """Depot-to-depot cost of one depot-free route."""

    if not route:
        return 0.0
    cost = edge_cost(instance, 0, route[0])
    cost += sum(edge_cost(instance, a, b) for a, b in pairwise(route))
    cost += edge_cost(instance, route[-1], 0)
    return float(cost)


def solution_cost(instance: VRPInstance, solution: Solution) -> float:
    return float(sum(route_cost(instance, route) for route in solution.routes))


def evaluate_solution(
    instance: VRPInstance,
    solution: Solution,
    *,
    reference_cost: float | None = None,
    runtime_s: float | None = None,
    peak_gpu_mb: float | None = None,
) -> ScaleMetrics:
    """Evaluate cost and feasibility of a depot-free solution."""

    visited = solution.visited_customers()
    counts = Counter(visited)
    expected = set(instance.customers)
    missing = len(expected.difference(counts))
    duplicates = sum(v - 1 for v in counts.values() if v > 1)
    capacity_violations = sum(
        _route_load(instance, route) > instance.capacity + 1e-9 for route in solution.routes
    )
    tw_violations = sum(_route_time_window_violations(instance, route) for route in solution.routes)
    cost = solution_cost(instance, solution)
    gap = None
    if reference_cost is not None and reference_cost > 0:
        gap = 100.0 * (cost - reference_cost) / reference_cost
    cost_per_customer = cost / max(instance.num_customers, 1)
    cost_per_route = cost / max(solution.num_routes, 1)
    routes_per_customer = solution.num_routes / max(instance.num_customers, 1)
    feasible = missing == 0 and duplicates == 0 and capacity_violations == 0 and tw_violations == 0
    return ScaleMetrics(
        cost=cost,
        feasible=bool(feasible),
        capacity_violations=int(capacity_violations),
        time_window_violations=int(tw_violations),
        missing_customers=missing,
        duplicate_customers=duplicates,
        num_routes=solution.num_routes,
        cost_per_customer=cost_per_customer,
        cost_per_route=cost_per_route,
        routes_per_customer=routes_per_customer,
        reference_cost=reference_cost,
        gap_to_reference=gap,
        runtime_s=runtime_s,
        peak_gpu_mb=peak_gpu_mb,
    )


def _route_load(instance: VRPInstance, route: tuple[int, ...]) -> float:
    return float(sum(float(instance.demand[c]) for c in route))


def _route_time_window_violations(instance: VRPInstance, route: tuple[int, ...]) -> int:
    if instance.tw_early is None or instance.tw_late is None:
        return 0
    service = instance.service_time
    time = 0.0
    current = 0
    violations = 0
    for customer in route:
        time += edge_cost(instance, current, customer)
        time = max(time, float(instance.tw_early[customer]))
        if time > float(instance.tw_late[customer]) + 1e-9:
            violations += 1
        if service is not None:
            time += float(service[customer])
        current = customer
    return violations
