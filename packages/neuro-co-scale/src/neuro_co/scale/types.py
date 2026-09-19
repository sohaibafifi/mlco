"""Shared data structures for large-scale VRP experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class VRPInstance:
    """Depot-first CVRP/CVRPTW-style instance.

    Customer ids are global integer ids in ``1..num_customers``. Depot id is 0.
    Routes in this package store only customer ids; depot legs are implicit.
    """

    coords: np.ndarray
    demand: np.ndarray
    capacity: float
    tw_early: np.ndarray | None = None
    tw_late: np.ndarray | None = None
    service_time: np.ndarray | None = None
    cost_matrix: np.ndarray | None = None
    round_distances: bool = False  # CVRPLIB EUC_2D: distances rounded to the nearest integer
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coords = np.asarray(self.coords, dtype=np.float64)
        demand = np.asarray(self.demand, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must have shape [n, 2], got {coords.shape}")
        if demand.shape != (coords.shape[0],):
            raise ValueError(f"demand must have shape [{coords.shape[0]}], got {demand.shape}")
        if coords.shape[0] < 2:
            raise ValueError("instance must contain a depot and at least one customer")
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}")
        _check_optional_vector("tw_early", self.tw_early, coords.shape[0])
        _check_optional_vector("tw_late", self.tw_late, coords.shape[0])
        _check_optional_vector("service_time", self.service_time, coords.shape[0])
        if self.cost_matrix is not None:
            cm = np.asarray(self.cost_matrix, dtype=np.float64)
            if cm.shape != (coords.shape[0], coords.shape[0]):
                raise ValueError(
                    f"cost_matrix must have shape [{coords.shape[0]}, {coords.shape[0]}], "
                    f"got {cm.shape}"
                )
            object.__setattr__(self, "cost_matrix", cm)
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "demand", demand)
        if self.tw_early is not None:
            object.__setattr__(self, "tw_early", np.asarray(self.tw_early, dtype=np.float64))
        if self.tw_late is not None:
            object.__setattr__(self, "tw_late", np.asarray(self.tw_late, dtype=np.float64))
        if self.service_time is not None:
            object.__setattr__(
                self, "service_time", np.asarray(self.service_time, dtype=np.float64)
            )

    @property
    def num_nodes(self) -> int:
        return int(self.coords.shape[0])

    @property
    def num_customers(self) -> int:
        return self.num_nodes - 1

    @property
    def customers(self) -> tuple[int, ...]:
        return tuple(range(1, self.num_nodes))

    def subinstance(self, customers: list[int] | tuple[int, ...]) -> tuple[VRPInstance, np.ndarray]:
        """Return a depot-first subinstance and its local-to-global id map."""

        ids = np.asarray([0, *customers], dtype=np.int64)
        return (
            VRPInstance(
                coords=self.coords[ids],
                demand=self.demand[ids],
                capacity=self.capacity,
                tw_early=None if self.tw_early is None else self.tw_early[ids],
                tw_late=None if self.tw_late is None else self.tw_late[ids],
                service_time=None if self.service_time is None else self.service_time[ids],
                cost_matrix=None
                if self.cost_matrix is None
                else self.cost_matrix[np.ix_(ids, ids)],
                name=f"{self.name}:sub" if self.name else "subinstance",
                metadata={**self.metadata, "global_ids": ids.tolist()},
            ),
            ids,
        )


@dataclass(frozen=True, slots=True)
class Partition:
    """Customer clusters over one parent instance."""

    clusters: tuple[tuple[int, ...], ...]
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_clusters(self) -> int:
        return len(self.clusters)

    def flattened(self) -> tuple[int, ...]:
        return tuple(c for cluster in self.clusters for c in cluster)


@dataclass(frozen=True, slots=True)
class Solution:
    """Depot-free routes over global customer ids."""

    routes: tuple[tuple[int, ...], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_routes(self) -> int:
        return len(self.routes)

    def visited_customers(self) -> tuple[int, ...]:
        return tuple(c for route in self.routes for c in route)


@dataclass(frozen=True, slots=True)
class ScaleMetrics:
    """Metrics used by the scaling benchmark."""

    cost: float
    feasible: bool
    capacity_violations: int
    time_window_violations: int
    missing_customers: int
    duplicate_customers: int
    num_routes: int
    cost_per_customer: float
    cost_per_route: float
    routes_per_customer: float
    reference_cost: float | None = None
    gap_to_reference: float | None = None
    runtime_s: float | None = None
    peak_gpu_mb: float | None = None


@dataclass(frozen=True, slots=True)
class PartitionQualityMetrics:
    """Pre-solve quality metrics for a customer partition."""

    num_clusters: int
    min_cluster_size: int
    max_cluster_size: int
    mean_cluster_size: float
    cluster_size_cv: float
    min_load_ratio: float
    max_load_ratio: float
    mean_load_ratio: float
    load_ratio_cv: float
    load_lower_bound_clusters: int
    cluster_count_ratio_to_load_bound: float
    clusters_per_customer: float
    overloaded_clusters: int
    missing_customers: int
    duplicate_customers: int
    mean_spatial_compactness: float
    mean_compatibility_distance: float
    mean_compatibility_score: float
    time_window_disjoint_pairs: int


def _check_optional_vector(name: str, value: np.ndarray | None, n: int) -> None:
    if value is None:
        return
    arr = np.asarray(value)
    if arr.shape != (n,):
        raise ValueError(f"{name} must have shape [{n}], got {arr.shape}")
