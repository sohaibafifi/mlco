"""Pairwise routing compatibility scores for VRP decomposition."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .features import client_feature_matrix, time_horizon
from .types import VRPInstance


@dataclass(frozen=True, slots=True)
class CompatibilityWeights:
    """Weights for pairwise routing distance components."""

    spatial: float = 1.0
    depot_distance: float = 0.25
    depot_angle: float = 0.25
    demand: float = 0.50
    time_window: float = 1.0
    service_time: float = 0.10

    def asdict(self) -> dict[str, float]:
        return asdict(self)


def pairwise_feature_distance(
    features: np.ndarray,
    *,
    feature_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return normalized Euclidean distances between feature rows."""

    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"features must have shape [n, d], got {matrix.shape}")
    if feature_weights is not None:
        weights = np.asarray(feature_weights, dtype=np.float64)
        if weights.shape != (matrix.shape[1],):
            raise ValueError(
                f"feature_weights must have shape [{matrix.shape[1]}], got {weights.shape}"
            )
        if np.any(weights < 0):
            raise ValueError("feature_weights must be non-negative")
        matrix = matrix * np.sqrt(weights)

    diff = matrix[:, None, :] - matrix[None, :, :]
    distance = np.linalg.norm(diff, axis=-1)
    max_distance = float(distance.max(initial=0.0))
    if max_distance > 0:
        distance = distance / max_distance
    np.fill_diagonal(distance, 0.0)
    return np.nan_to_num(distance, copy=False)


def routing_compatibility_distance_matrix(
    instance: VRPInstance,
    weights: CompatibilityWeights | None = None,
) -> np.ndarray:
    """Return customer-only pairwise distances for routing compatibility.

    Lower values mean that two customers are more compatible inside the same
    decomposed subproblem.
    """

    w = CompatibilityWeights() if weights is None else weights
    customers = np.asarray(instance.customers, dtype=np.int64)
    coords = instance.coords[customers]
    distance = np.zeros((instance.num_customers, instance.num_customers), dtype=np.float64)
    total_weight = 0.0

    def add(component: np.ndarray, weight: float) -> None:
        nonlocal distance, total_weight
        if weight < 0:
            raise ValueError("compatibility weights must be non-negative")
        if weight == 0:
            return
        distance += weight * component
        total_weight += weight

    if w.spatial:
        add(_normalized_pairwise_euclidean(coords), w.spatial)

    depot = instance.coords[0]
    deltas = coords - depot
    depot_distance = np.linalg.norm(deltas, axis=1)
    if w.depot_distance:
        add(_normalized_outer_abs(depot_distance), w.depot_distance)

    if w.depot_angle:
        angle = np.arctan2(deltas[:, 1], deltas[:, 0])
        add(_normalized_pairwise_angle(angle), w.depot_angle)

    if w.demand:
        demand_ratio = instance.demand[customers] / instance.capacity
        demand_pair = demand_ratio[:, None] + demand_ratio[None, :]
        add(np.minimum(demand_pair, 2.0) / 2.0, w.demand)

    if w.time_window and instance.tw_early is not None and instance.tw_late is not None:
        add(_time_window_distance(instance, customers), w.time_window)

    if w.service_time and instance.service_time is not None:
        add(_normalized_outer_abs(instance.service_time[customers]), w.service_time)

    if total_weight == 0:
        features, _ = client_feature_matrix(instance)
        distance = pairwise_feature_distance(features)
    else:
        distance = distance / total_weight
    np.fill_diagonal(distance, 0.0)
    return np.nan_to_num(distance, copy=False)


def routing_compatibility_score_matrix(
    instance: VRPInstance,
    weights: CompatibilityWeights | None = None,
) -> np.ndarray:
    """Return pairwise compatibility scores in (0, 1]."""

    distance = routing_compatibility_distance_matrix(instance, weights)
    score = 1.0 / (1.0 + distance)
    np.fill_diagonal(score, 1.0)
    return score


def _normalized_pairwise_euclidean(values: np.ndarray) -> np.ndarray:
    diff = values[:, None, :] - values[None, :, :]
    distance = np.linalg.norm(diff, axis=-1)
    max_distance = float(distance.max(initial=0.0))
    if max_distance > 0:
        distance = distance / max_distance
    np.fill_diagonal(distance, 0.0)
    return distance


def _normalized_outer_abs(values: np.ndarray) -> np.ndarray:
    distance = np.abs(values[:, None] - values[None, :])
    max_distance = float(distance.max(initial=0.0))
    if max_distance > 0:
        distance = distance / max_distance
    np.fill_diagonal(distance, 0.0)
    return distance


def _normalized_pairwise_angle(angle: np.ndarray) -> np.ndarray:
    diff = np.abs(angle[:, None] - angle[None, :])
    diff = np.minimum(diff, 2 * np.pi - diff)
    distance = diff / np.pi
    np.fill_diagonal(distance, 0.0)
    return distance


def _time_window_distance(instance: VRPInstance, customers: np.ndarray) -> np.ndarray:
    assert instance.tw_early is not None
    assert instance.tw_late is not None
    horizon = time_horizon(instance)
    early = instance.tw_early[customers]
    late = instance.tw_late[customers]
    width = np.maximum(late - early, 0.0)
    center = 0.5 * (early + late)

    gap_ij = np.maximum(early[:, None] - late[None, :], 0.0)
    gap_ji = np.maximum(early[None, :] - late[:, None], 0.0)
    gap = np.maximum(gap_ij, gap_ji) / horizon
    center_distance = np.abs(center[:, None] - center[None, :]) / horizon
    width_distance = np.abs(width[:, None] - width[None, :]) / horizon
    distance = gap + 0.25 * center_distance + 0.10 * width_distance
    np.fill_diagonal(distance, 0.0)
    return distance
