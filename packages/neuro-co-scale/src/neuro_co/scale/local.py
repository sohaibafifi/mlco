"""Local subproblem interfaces for bounded cluster solving."""

from __future__ import annotations

from dataclasses import dataclass

from .types import Partition, Solution, VRPInstance


@dataclass(frozen=True, slots=True)
class LocalSubproblem:
    """Depot-first bounded VRP subproblem extracted from one cluster."""

    cluster_index: int
    instance: VRPInstance
    local_to_global: tuple[int, ...]

    @property
    def global_customers(self) -> tuple[int, ...]:
        return self.local_to_global[1:]

    @property
    def max_customer_id(self) -> int:
        return len(self.local_to_global) - 1

    def to_global_route(self, local_route: tuple[int, ...]) -> tuple[int, ...]:
        """Map a depot-free local route to global customer ids."""

        return tuple(self.local_to_global[customer] for customer in local_route)

    def to_global_solution(self, local_solution: Solution) -> Solution:
        """Map depot-free local routes to a global solution fragment."""

        return Solution(
            routes=tuple(self.to_global_route(route) for route in local_solution.routes),
            metadata={
                **local_solution.metadata,
                "cluster_index": self.cluster_index,
                "global_customers": list(self.global_customers),
            },
        )


def extract_local_subproblem(
    instance: VRPInstance,
    customers: tuple[int, ...],
    *,
    cluster_index: int = 0,
) -> LocalSubproblem:
    """Extract `depot + customers` as a bounded local VRP subproblem."""

    if not customers:
        raise ValueError("customers must not be empty")
    _validate_customers(instance, customers)
    local_instance, local_to_global = instance.subinstance(customers)
    return LocalSubproblem(
        cluster_index=cluster_index,
        instance=local_instance,
        local_to_global=tuple(int(idx) for idx in local_to_global),
    )


def iter_local_subproblems(
    instance: VRPInstance,
    partition: Partition,
) -> tuple[LocalSubproblem, ...]:
    """Extract all non-empty partition clusters as local subproblems."""

    return tuple(
        extract_local_subproblem(instance, cluster, cluster_index=idx)
        for idx, cluster in enumerate(partition.clusters)
        if cluster
    )


def merge_local_solutions(
    subproblems: tuple[LocalSubproblem, ...],
    local_solutions: tuple[Solution, ...],
) -> Solution:
    """Merge local solution fragments into one global depot-free solution."""

    if len(subproblems) != len(local_solutions):
        raise ValueError("subproblems and local_solutions must have the same length")
    routes: list[tuple[int, ...]] = []
    for subproblem, local_solution in zip(subproblems, local_solutions, strict=True):
        routes.extend(subproblem.to_global_solution(local_solution).routes)
    return Solution(routes=tuple(routes), metadata={"source": "local_subproblems"})


def _validate_customers(instance: VRPInstance, customers: tuple[int, ...]) -> None:
    seen: set[int] = set()
    for customer in customers:
        if customer == 0:
            raise ValueError("customers must not include depot id 0")
        if customer < 0 or customer >= instance.num_nodes:
            raise ValueError(f"customer id {customer} is outside the instance")
        if customer in seen:
            raise ValueError(f"duplicate customer id {customer}")
        seen.add(customer)
