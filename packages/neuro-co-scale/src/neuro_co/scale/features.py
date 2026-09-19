"""Feature extraction for VRP-aware decomposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import VRPInstance


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Controls which client features enter a decomposition model."""

    include_coordinates: bool = True
    include_depot_distance: bool = True
    include_depot_angle: bool = True
    include_demand: bool = True
    include_time_windows: bool = True
    include_service_time: bool = True
    normalize_coordinates: bool = True


def client_feature_matrix(
    instance: VRPInstance,
    config: FeatureConfig | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return a customer-only feature matrix and feature names.

    Rows follow global customer ids in ascending order. Depot id 0 is used to
    derive depot-relative features but is not returned as a row.
    """

    cfg = FeatureConfig() if config is None else config
    customers = np.asarray(instance.customers, dtype=np.int64)
    coords = instance.coords[customers]
    parts: list[np.ndarray] = []
    names: list[str] = []

    if cfg.include_coordinates:
        xy = _normalized_coordinates(instance, coords) if cfg.normalize_coordinates else coords
        parts.append(xy)
        names.extend(["x", "y"])

    depot = instance.coords[0]
    deltas = coords - depot
    depot_distance = np.linalg.norm(deltas, axis=1)
    max_depot_distance = max(float(depot_distance.max(initial=0.0)), 1e-12)

    if cfg.include_depot_distance:
        parts.append((depot_distance / max_depot_distance)[:, None])
        names.append("depot_distance")

    if cfg.include_depot_angle:
        angle = np.arctan2(deltas[:, 1], deltas[:, 0])
        parts.append(np.column_stack([np.sin(angle), np.cos(angle)]))
        names.extend(["depot_angle_sin", "depot_angle_cos"])

    if cfg.include_demand:
        parts.append((instance.demand[customers] / instance.capacity)[:, None])
        names.append("demand_ratio")

    horizon = time_horizon(instance)
    if cfg.include_time_windows:
        tw_early = (
            np.zeros(instance.num_customers, dtype=np.float64)
            if instance.tw_early is None
            else instance.tw_early[customers]
        )
        tw_late = (
            np.full(instance.num_customers, horizon, dtype=np.float64)
            if instance.tw_late is None
            else instance.tw_late[customers]
        )
        parts.append(
            np.column_stack(
                [
                    tw_early / horizon,
                    tw_late / horizon,
                    np.maximum(tw_late - tw_early, 0.0) / horizon,
                ]
            )
        )
        names.extend(["tw_early", "tw_late", "tw_width"])

    if cfg.include_service_time:
        service_time = (
            np.zeros(instance.num_customers, dtype=np.float64)
            if instance.service_time is None
            else instance.service_time[customers]
        )
        parts.append((service_time / horizon)[:, None])
        names.append("service_time")

    if not parts:
        raise ValueError("at least one feature family must be enabled")
    matrix = np.column_stack(parts).astype(np.float64, copy=False)
    return np.nan_to_num(matrix, copy=False), tuple(names)


def time_horizon(instance: VRPInstance) -> float:
    """Return a stable normalizer for time-window and service-time features."""

    candidates = [1.0]
    for values in (instance.tw_early, instance.tw_late, instance.service_time):
        if values is not None and values.size:
            finite = np.asarray(values, dtype=np.float64)[np.isfinite(values)]
            if finite.size:
                candidates.append(float(finite.max()))
    return max(max(candidates), 1e-12)


def _normalized_coordinates(instance: VRPInstance, coords: np.ndarray) -> np.ndarray:
    mins = instance.coords.min(axis=0)
    span = np.maximum(instance.coords.max(axis=0) - mins, 1e-12)
    return (coords - mins) / span
