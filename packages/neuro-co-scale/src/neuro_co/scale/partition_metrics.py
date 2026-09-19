"""Pre-solve quality metrics for VRP decompositions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from math import ceil

import numpy as np

from .compatibility import CompatibilityWeights
from .features import time_horizon
from .types import Partition, PartitionQualityMetrics, VRPInstance


@dataclass(frozen=True, slots=True)
class _CompatibilityContext:
    weights: CompatibilityWeights
    max_spatial_distance: float
    depot_distance: np.ndarray
    max_depot_distance_delta: float
    depot_angle: np.ndarray
    demand_ratio: np.ndarray
    horizon: float
    service_time: np.ndarray | None
    max_service_delta: float


def evaluate_partition(
    instance: VRPInstance,
    partition: Partition,
    *,
    max_demand: float | None = None,
    weights: CompatibilityWeights | None = None,
) -> PartitionQualityMetrics:
    """Evaluate a partition before local solving and recomposition."""

    clusters = [tuple(cluster) for cluster in partition.clusters if cluster]
    sizes = np.asarray([len(cluster) for cluster in clusters], dtype=np.float64)
    load_budget = float(instance.capacity if max_demand is None else max_demand)
    loads = np.asarray([_cluster_load(instance, cluster) for cluster in clusters], dtype=np.float64)
    load_ratios = loads / max(load_budget, 1e-12)
    load_lower_bound = max(
        1,
        ceil(float(instance.demand[list(instance.customers)].sum()) / max(load_budget, 1e-12)),
    )
    counts = Counter(partition.flattened())
    expected = set(instance.customers)
    missing = len(expected.difference(counts))
    duplicates = sum(v - 1 for v in counts.values() if v > 1)

    spatial_values, compatibility_distance_values, compatibility_score_values = (
        _cluster_pair_metrics(instance, clusters, weights)
    )

    return PartitionQualityMetrics(
        num_clusters=len(clusters),
        min_cluster_size=int(_min_or_zero(sizes)),
        max_cluster_size=int(_max_or_zero(sizes)),
        mean_cluster_size=float(sizes.mean()) if sizes.size else 0.0,
        cluster_size_cv=_coefficient_of_variation(sizes),
        min_load_ratio=_min_or_zero(load_ratios),
        max_load_ratio=_max_or_zero(load_ratios),
        mean_load_ratio=float(load_ratios.mean()) if load_ratios.size else 0.0,
        load_ratio_cv=_coefficient_of_variation(load_ratios),
        load_lower_bound_clusters=load_lower_bound,
        cluster_count_ratio_to_load_bound=len(clusters) / load_lower_bound,
        clusters_per_customer=len(clusters) / max(instance.num_customers, 1),
        overloaded_clusters=int(np.sum(loads > load_budget + 1e-9)),
        missing_customers=missing,
        duplicate_customers=duplicates,
        mean_spatial_compactness=_mean_or_zero(spatial_values),
        mean_compatibility_distance=_mean_or_zero(compatibility_distance_values),
        mean_compatibility_score=_mean_or_zero(compatibility_score_values),
        time_window_disjoint_pairs=_time_window_disjoint_pairs(instance, clusters),
    )


def _cluster_load(instance: VRPInstance, cluster: tuple[int, ...]) -> float:
    return float(sum(float(instance.demand[customer]) for customer in cluster))


def _cluster_pair_metrics(
    instance: VRPInstance,
    clusters: list[tuple[int, ...]],
    weights: CompatibilityWeights | None,
) -> tuple[list[float], list[float], list[float]]:
    ctx = _compatibility_context(instance, weights)
    spatial_values: list[float] = []
    compatibility_distance_values: list[float] = []
    compatibility_score_values: list[float] = []
    for cluster in clusters:
        if len(cluster) < 2:
            spatial_values.append(0.0)
            compatibility_distance_values.append(0.0)
            compatibility_score_values.append(0.0)
            continue
        spatial_pair_values: list[float] = []
        compatibility_pair_values: list[float] = []
        score_pair_values: list[float] = []
        for left, right in combinations(cluster, 2):
            spatial = _pair_spatial_distance(instance, left, right, ctx)
            compatibility = _pair_compatibility_distance(instance, left, right, ctx, spatial)
            spatial_pair_values.append(spatial)
            compatibility_pair_values.append(compatibility)
            score_pair_values.append(1.0 / (1.0 + compatibility))
        spatial_values.append(float(np.mean(spatial_pair_values)))
        compatibility_distance_values.append(float(np.mean(compatibility_pair_values)))
        compatibility_score_values.append(float(np.mean(score_pair_values)))
    return spatial_values, compatibility_distance_values, compatibility_score_values


def _compatibility_context(
    instance: VRPInstance,
    weights: CompatibilityWeights | None,
) -> _CompatibilityContext:
    customers = np.asarray(instance.customers, dtype=np.int64)
    coords = instance.coords[customers]
    coord_span = coords.max(axis=0) - coords.min(axis=0)
    max_spatial_distance = max(float(np.linalg.norm(coord_span)), 1e-12)

    deltas = coords - instance.coords[0]
    depot_distance_values = np.linalg.norm(deltas, axis=1)
    depot_distance = np.zeros(instance.num_nodes, dtype=np.float64)
    depot_distance[customers] = depot_distance_values
    max_depot_distance_delta = max(
        float(depot_distance_values.max(initial=0.0) - depot_distance_values.min(initial=0.0)),
        1e-12,
    )

    depot_angle = np.zeros(instance.num_nodes, dtype=np.float64)
    depot_angle[customers] = np.arctan2(deltas[:, 1], deltas[:, 0])

    demand_ratio = np.zeros(instance.num_nodes, dtype=np.float64)
    demand_ratio[customers] = instance.demand[customers] / instance.capacity

    max_service_delta = 1e-12
    if instance.service_time is not None:
        service_values = instance.service_time[customers]
        max_service_delta = max(
            float(service_values.max(initial=0.0) - service_values.min(initial=0.0)),
            1e-12,
        )

    return _CompatibilityContext(
        weights=CompatibilityWeights() if weights is None else weights,
        max_spatial_distance=max_spatial_distance,
        depot_distance=depot_distance,
        max_depot_distance_delta=max_depot_distance_delta,
        depot_angle=depot_angle,
        demand_ratio=demand_ratio,
        horizon=time_horizon(instance),
        service_time=instance.service_time,
        max_service_delta=max_service_delta,
    )


def _pair_spatial_distance(
    instance: VRPInstance,
    left: int,
    right: int,
    ctx: _CompatibilityContext,
) -> float:
    diff = instance.coords[left] - instance.coords[right]
    return float(np.sqrt(np.dot(diff, diff)) / ctx.max_spatial_distance)


def _pair_compatibility_distance(
    instance: VRPInstance,
    left: int,
    right: int,
    ctx: _CompatibilityContext,
    spatial_distance: float,
) -> float:
    distance = 0.0
    total_weight = 0.0

    def add(component: float, weight: float) -> None:
        nonlocal distance, total_weight
        if weight < 0:
            raise ValueError("compatibility weights must be non-negative")
        if weight == 0:
            return
        distance += weight * component
        total_weight += weight

    w = ctx.weights
    add(spatial_distance, w.spatial)
    add(
        abs(ctx.depot_distance[left] - ctx.depot_distance[right]) / ctx.max_depot_distance_delta,
        w.depot_distance,
    )
    angle_diff = abs(ctx.depot_angle[left] - ctx.depot_angle[right])
    angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
    add(float(angle_diff / np.pi), w.depot_angle)
    demand_pair = ctx.demand_ratio[left] + ctx.demand_ratio[right]
    add(float(min(demand_pair, 2.0) / 2.0), w.demand)
    if instance.tw_early is not None and instance.tw_late is not None:
        add(_time_window_pair_distance(instance, left, right, ctx.horizon), w.time_window)
    if ctx.service_time is not None:
        add(
            float(abs(ctx.service_time[left] - ctx.service_time[right]) / ctx.max_service_delta),
            w.service_time,
        )
    if total_weight == 0:
        return spatial_distance
    return float(distance / total_weight)


def _time_window_pair_distance(
    instance: VRPInstance,
    left: int,
    right: int,
    horizon: float,
) -> float:
    assert instance.tw_early is not None
    assert instance.tw_late is not None
    left_early = float(instance.tw_early[left])
    left_late = float(instance.tw_late[left])
    right_early = float(instance.tw_early[right])
    right_late = float(instance.tw_late[right])
    left_width = max(left_late - left_early, 0.0)
    right_width = max(right_late - right_early, 0.0)
    gap = max(max(left_early - right_late, 0.0), max(right_early - left_late, 0.0))
    center_distance = abs(0.5 * (left_early + left_late - right_early - right_late))
    width_distance = abs(left_width - right_width)
    return float((gap + 0.25 * center_distance + 0.10 * width_distance) / horizon)


def _time_window_disjoint_pairs(instance: VRPInstance, clusters: list[tuple[int, ...]]) -> int:
    if instance.tw_early is None or instance.tw_late is None:
        return 0
    count = 0
    for cluster in clusters:
        for left, right in combinations(cluster, 2):
            left_early = float(instance.tw_early[left])
            left_late = float(instance.tw_late[left])
            right_early = float(instance.tw_early[right])
            right_late = float(instance.tw_late[right])
            if left_late < right_early or right_late < left_early:
                count += 1
    return count


def _coefficient_of_variation(values: np.ndarray) -> float:
    if not values.size:
        return 0.0
    mean = float(values.mean())
    if abs(mean) <= 1e-12:
        return 0.0
    return float(values.std() / mean)


def _mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _min_or_zero(values: np.ndarray) -> float:
    if not values.size:
        return 0.0
    return float(values.min())


def _max_or_zero(values: np.ndarray) -> float:
    if not values.size:
        return 0.0
    return float(values.max())
