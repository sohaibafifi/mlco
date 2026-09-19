"""Partitioners for large CVRP and CVRPTW instances."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import atan2, ceil
from typing import Literal

import numpy as np

from .compatibility import CompatibilityWeights, routing_compatibility_distance_matrix
from .features import client_feature_matrix, time_horizon
from .types import Partition, VRPInstance

RefinementScoreMode = Literal["compactness", "route", "hybrid"]


@dataclass(frozen=True, slots=True)
class _Relocation:
    target_idx: int
    gain: float
    new_source_score: float
    new_target_score: float
    new_source_route_score: float | None = None
    new_target_route_score: float | None = None


def sweep_partition(
    instance: VRPInstance,
    *,
    max_customers: int,
    max_demand: float | None = None,
    start_angle: float = 0.0,
) -> Partition:
    """Angular sweep partition around the depot.

    The method is deterministic and intentionally simple. It is a baseline for
    later learned or constraint-aware partitioners.
    """

    if max_customers <= 0:
        raise ValueError("max_customers must be positive")
    depot = instance.coords[0]
    max_load = float(instance.capacity if max_demand is None else max_demand)
    ordered = sorted(
        instance.customers,
        key=lambda idx: _wrapped_angle(instance.coords[idx], depot, start_angle),
    )
    clusters = _split_ordered(instance, ordered, max_customers=max_customers, max_demand=max_load)
    return Partition(
        clusters=clusters,
        method="sweep",
        metadata={"max_customers": max_customers, "max_demand": max_load},
    )


def capacity_aware_sweep_partition(
    instance: VRPInstance,
    *,
    max_customers: int,
    capacity_fraction: float = 0.95,
    start_angle: float = 0.0,
) -> Partition:
    """Sweep partition with a conservative capacity budget per cluster."""

    if not 0 < capacity_fraction <= 1:
        raise ValueError("capacity_fraction must be in (0, 1]")
    return sweep_partition(
        instance,
        max_customers=max_customers,
        max_demand=instance.capacity * capacity_fraction,
        start_angle=start_angle,
    )


def grid_partition(
    instance: VRPInstance,
    *,
    max_customers: int,
    grid_shape: tuple[int, int] | None = None,
    max_demand: float | None = None,
) -> Partition:
    """Partition customers by spatial grid cells, then split oversized cells."""

    if max_customers <= 0:
        raise ValueError("max_customers must be positive")
    coords = instance.coords[1:]
    if grid_shape is None:
        side = max(1, ceil((instance.num_customers / max_customers) ** 0.5))
        grid_shape = (side, side)
    gx, gy = grid_shape
    if gx <= 0 or gy <= 0:
        raise ValueError(f"grid_shape must be positive, got {grid_shape}")

    mins = coords.min(axis=0)
    span = np.maximum(coords.max(axis=0) - mins, 1e-12)
    cell_ids: dict[tuple[int, int], list[int]] = defaultdict(list)
    for customer in instance.customers:
        rel = (instance.coords[customer] - mins) / span
        ix = min(gx - 1, max(0, int(rel[0] * gx)))
        iy = min(gy - 1, max(0, int(rel[1] * gy)))
        cell_ids[(ix, iy)].append(customer)

    clusters: list[tuple[int, ...]] = []
    max_load = float(instance.capacity if max_demand is None else max_demand)
    for cell in sorted(cell_ids):
        ordered = sorted(
            cell_ids[cell], key=lambda idx: (instance.coords[idx][0], instance.coords[idx][1])
        )
        clusters.extend(
            _split_ordered(
                instance,
                ordered,
                max_customers=max_customers,
                max_demand=max_load,
            )
        )
    return Partition(
        clusters=tuple(c for c in clusters if c),
        method="grid",
        metadata={"max_customers": max_customers, "max_demand": max_load, "grid_shape": grid_shape},
    )


def morton_order(instance: VRPInstance, *, bits: int = 16) -> tuple[int, ...]:
    """Order customers by Morton Z-order keys in normalized coordinate space."""

    if bits <= 0 or bits > 31:
        raise ValueError("bits must be in [1, 31]")
    customers = np.asarray(instance.customers, dtype=np.int64)
    coords = instance.coords[customers]
    mins = coords.min(axis=0)
    span = np.maximum(coords.max(axis=0) - mins, 1e-12)
    scale = float((1 << bits) - 1)
    quantized = np.clip(((coords - mins) / span) * scale, 0.0, scale).astype(np.uint64)
    keys = _morton_codes(quantized[:, 0], quantized[:, 1])
    order = np.lexsort((customers, quantized[:, 1], quantized[:, 0], keys))
    return tuple(int(customers[idx]) for idx in order)


def morton_partition(
    instance: VRPInstance,
    *,
    max_customers: int,
    max_demand: float | None = None,
    capacity_fraction: float = 0.95,
    bits: int = 16,
) -> Partition:
    """Partition customers by Morton Z-order, then split by size and capacity."""

    if max_customers <= 0:
        raise ValueError("max_customers must be positive")
    if not 0 < capacity_fraction <= 1:
        raise ValueError("capacity_fraction must be in (0, 1]")
    max_load = float(instance.capacity * capacity_fraction if max_demand is None else max_demand)
    if max_load <= 0:
        raise ValueError("max_demand must be positive")
    ordered = list(morton_order(instance, bits=bits))
    clusters = _split_ordered(instance, ordered, max_customers=max_customers, max_demand=max_load)
    return Partition(
        clusters=clusters,
        method="morton",
        metadata={
            "max_customers": max_customers,
            "max_demand": max_load,
            "capacity_fraction": capacity_fraction,
            "bits": bits,
            "complexity": "O(n log n) sort, O(n) memory",
        },
    )


def morton_refined_partition(
    instance: VRPInstance,
    *,
    max_customers: int,
    max_demand: float | None = None,
    capacity_fraction: float = 0.95,
    bits: int = 16,
    boundary_customers: int = 8,
    neighbor_span: int = 1,
    max_passes: int = 1,
    min_gain: float = 1e-9,
    score_mode: RefinementScoreMode = "hybrid",
    hybrid_shortlist: int = 2,
) -> Partition:
    """Morton partition followed by sparse boundary relocation.

    The refinement only considers a bounded number of customers at the two ends
    of each Morton cluster and only tests nearby clusters in Morton order. It
    avoids dense pairwise matrices and keeps memory linear in the number of
    customers.
    """

    if boundary_customers <= 0:
        raise ValueError("boundary_customers must be positive")
    if neighbor_span <= 0:
        raise ValueError("neighbor_span must be positive")
    if max_passes <= 0:
        raise ValueError("max_passes must be positive")
    if score_mode not in ("compactness", "route", "hybrid"):
        raise ValueError("score_mode must be 'compactness', 'route', or 'hybrid'")
    if hybrid_shortlist <= 0:
        raise ValueError("hybrid_shortlist must be positive")

    base = morton_partition(
        instance,
        max_customers=max_customers,
        max_demand=max_demand,
        capacity_fraction=capacity_fraction,
        bits=bits,
    )
    order = morton_order(instance, bits=bits)
    rank = {customer: pos for pos, customer in enumerate(order)}
    clusters = [list(cluster) for cluster in base.clusters]
    loads = [float(instance.demand[cluster].sum()) if cluster else 0.0 for cluster in clusters]
    scores = [_cluster_score(instance, cluster, score_mode=score_mode) for cluster in clusters]
    route_scores = None
    if score_mode == "hybrid":
        route_scores = [_cluster_route_proxy(instance, cluster) for cluster in clusters]
    max_load = float(base.metadata["max_demand"])

    moves = 0
    for _ in range(max_passes):
        moved_this_pass = _refine_morton_boundary_once(
            instance,
            clusters,
            loads,
            scores,
            route_scores,
            rank=rank,
            max_customers=max_customers,
            max_load=max_load,
            boundary_customers=boundary_customers,
            neighbor_span=neighbor_span,
            min_gain=min_gain,
            score_mode=score_mode,
            hybrid_shortlist=hybrid_shortlist,
        )
        moves += moved_this_pass
        if moved_this_pass == 0:
            break

    refined = tuple(tuple(cluster) for cluster in clusters if cluster)
    return Partition(
        clusters=refined,
        method="morton_refined",
        metadata={
            **base.metadata,
            "base_method": base.method,
            "boundary_customers": boundary_customers,
            "neighbor_span": neighbor_span,
            "max_passes": max_passes,
            "min_gain": min_gain,
            "score_mode": score_mode,
            "hybrid_shortlist": hybrid_shortlist,
            "moves": moves,
            "refinement": "sparse_boundary_relocation",
            "complexity": _morton_refinement_complexity(score_mode),
        },
    )


def feature_aware_partition(
    instance: VRPInstance,
    *,
    max_customers: int,
    max_demand: float | None = None,
    capacity_fraction: float = 0.95,
    weights: CompatibilityWeights | None = None,
) -> Partition:
    """Partition customers with multi-feature routing compatibility."""

    if max_customers <= 0:
        raise ValueError("max_customers must be positive")
    if not 0 < capacity_fraction <= 1:
        raise ValueError("capacity_fraction must be in (0, 1]")
    max_load = float(instance.capacity * capacity_fraction if max_demand is None else max_demand)
    if max_load <= 0:
        raise ValueError("max_demand must be positive")

    customers = list(instance.customers)
    distance = routing_compatibility_distance_matrix(instance, weights)
    target_clusters = _target_cluster_count(
        instance, max_customers=max_customers, max_demand=max_load
    )
    seed_positions = _select_farthest_seed_positions(instance, distance, target_clusters)

    clusters: list[list[int]] = [[customers[pos]] for pos in seed_positions]
    loads = [float(instance.demand[customers[pos]]) for pos in seed_positions]
    assigned = set(seed_positions)

    for pos in _assignment_order(instance, distance):
        if pos in assigned:
            continue
        customer = customers[pos]
        demand = float(instance.demand[customer])
        candidate_scores: list[tuple[float, int]] = []
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) >= max_customers:
                continue
            if loads[cluster_idx] + demand > max_load:
                continue
            candidate_scores.append(
                (
                    _cluster_assignment_score(
                        distance,
                        pos,
                        cluster,
                        load=loads[cluster_idx],
                        max_customers=max_customers,
                        max_demand=max_load,
                    ),
                    cluster_idx,
                )
            )
        if not candidate_scores:
            clusters.append([customer])
            loads.append(demand)
        else:
            _, cluster_idx = min(candidate_scores)
            clusters[cluster_idx].append(customer)
            loads[cluster_idx] += demand
        assigned.add(pos)

    _, feature_names = client_feature_matrix(instance)
    used_weights = CompatibilityWeights() if weights is None else weights
    return Partition(
        clusters=tuple(tuple(cluster) for cluster in clusters if cluster),
        method="feature_aware",
        metadata={
            "max_customers": max_customers,
            "max_demand": max_load,
            "capacity_fraction": capacity_fraction,
            "weights": used_weights.asdict(),
            "feature_names": feature_names,
        },
    )


def _split_ordered(
    instance: VRPInstance,
    ordered: list[int],
    *,
    max_customers: int,
    max_demand: float,
) -> tuple[tuple[int, ...], ...]:
    clusters: list[tuple[int, ...]] = []
    current: list[int] = []
    load = 0.0
    for customer in ordered:
        demand = float(instance.demand[customer])
        too_many = len(current) >= max_customers
        too_loaded = current and load + demand > max_demand
        if too_many or too_loaded:
            clusters.append(tuple(current))
            current = []
            load = 0.0
        current.append(customer)
        load += demand
    if current:
        clusters.append(tuple(current))
    return tuple(clusters)


def _wrapped_angle(point: np.ndarray, depot: np.ndarray, start_angle: float) -> float:
    angle = atan2(float(point[1] - depot[1]), float(point[0] - depot[0])) - start_angle
    return angle % (2 * np.pi)


def _morton_codes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _part1by1(x) | (_part1by1(y) << np.uint64(1))


def _part1by1(values: np.ndarray) -> np.ndarray:
    x = values.astype(np.uint64, copy=True)
    x &= np.uint64(0x00000000FFFFFFFF)
    x = (x | (x << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x00FF00FF00FF00FF)
    x = (x | (x << np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = (x | (x << np.uint64(2))) & np.uint64(0x3333333333333333)
    return (x | (x << np.uint64(1))) & np.uint64(0x5555555555555555)


def _refine_morton_boundary_once(
    instance: VRPInstance,
    clusters: list[list[int]],
    loads: list[float],
    scores: list[float],
    route_scores: list[float] | None,
    *,
    rank: dict[int, int],
    max_customers: int,
    max_load: float,
    boundary_customers: int,
    neighbor_span: int,
    min_gain: float,
    score_mode: RefinementScoreMode,
    hybrid_shortlist: int,
) -> int:
    moved = 0
    moved_customers: set[int] = set()
    for source_idx in range(len(clusters)):
        for customer in _boundary_candidates(clusters[source_idx], boundary_customers):
            if customer in moved_customers or customer not in clusters[source_idx]:
                continue
            move = _best_boundary_relocation(
                instance,
                clusters,
                loads,
                scores,
                route_scores,
                source_idx,
                customer,
                rank=rank,
                max_customers=max_customers,
                max_load=max_load,
                neighbor_span=neighbor_span,
                min_gain=min_gain,
                score_mode=score_mode,
                hybrid_shortlist=hybrid_shortlist,
            )
            if move is None:
                continue
            _apply_boundary_relocation(
                clusters,
                loads,
                scores,
                route_scores,
                source_idx,
                move.target_idx,
                customer,
                rank=rank,
                demand=float(instance.demand[customer]),
                new_source_score=move.new_source_score,
                new_target_score=move.new_target_score,
                new_source_route_score=move.new_source_route_score,
                new_target_route_score=move.new_target_route_score,
            )
            moved_customers.add(customer)
            moved += 1
            if move.gain <= min_gain:
                raise AssertionError("accepted relocation must improve the local proxy")
    return moved


def _best_boundary_relocation(
    instance: VRPInstance,
    clusters: list[list[int]],
    loads: list[float],
    scores: list[float],
    route_scores: list[float] | None,
    source_idx: int,
    customer: int,
    *,
    rank: dict[int, int],
    max_customers: int,
    max_load: float,
    neighbor_span: int,
    min_gain: float,
    score_mode: RefinementScoreMode,
    hybrid_shortlist: int,
) -> _Relocation | None:
    source = clusters[source_idx]
    demand = float(instance.demand[customer])
    source_score = scores[source_idx]
    new_source = [c for c in source if c != customer]
    new_source_score = _cluster_score(instance, new_source, score_mode=score_mode)
    compactness_candidates: list[tuple[float, int, float, float]] = []
    best: _Relocation | None = None
    start = max(0, source_idx - neighbor_span)
    stop = min(len(clusters), source_idx + neighbor_span + 1)
    for target_idx in range(start, stop):
        if target_idx == source_idx:
            continue
        target = clusters[target_idx]
        if len(target) >= max_customers:
            continue
        if loads[target_idx] + demand > max_load + 1e-9:
            continue
        current_score = source_score + scores[target_idx]
        new_target = _with_customer_by_rank(target, customer, rank)
        new_target_score = _cluster_score(instance, new_target, score_mode=score_mode)
        new_score = new_source_score + new_target_score
        gain = current_score - new_score
        if gain <= min_gain:
            continue
        if score_mode == "hybrid":
            compactness_candidates.append((gain, target_idx, new_source_score, new_target_score))
            continue
        candidate = (gain, -abs(target_idx - source_idx), -target_idx)
        if best is None or candidate > (
            best.gain,
            -abs(best.target_idx - source_idx),
            -best.target_idx,
        ):
            best = _Relocation(
                target_idx=target_idx,
                gain=gain,
                new_source_score=new_source_score,
                new_target_score=new_target_score,
            )
    if score_mode != "hybrid":
        return best
    if route_scores is None:
        raise AssertionError("hybrid scoring requires route score cache")
    return _best_hybrid_boundary_relocation(
        instance,
        clusters,
        route_scores,
        source_idx,
        customer,
        candidates=compactness_candidates,
        min_gain=min_gain,
        hybrid_shortlist=hybrid_shortlist,
    )


def _best_hybrid_boundary_relocation(
    instance: VRPInstance,
    clusters: list[list[int]],
    route_scores: list[float],
    source_idx: int,
    customer: int,
    *,
    candidates: list[tuple[float, int, float, float]],
    min_gain: float,
    hybrid_shortlist: int,
) -> _Relocation | None:
    shortlist = sorted(
        candidates,
        key=lambda item: (item[0], -abs(item[1] - source_idx), -item[1]),
        reverse=True,
    )[:hybrid_shortlist]
    if not shortlist:
        return None

    source = clusters[source_idx]
    new_source = [c for c in source if c != customer]
    new_source_route_score = _cluster_route_proxy(instance, new_source)
    best: _Relocation | None = None
    for _, target_idx, new_source_score, new_target_score in shortlist:
        current_route_score = route_scores[source_idx] + route_scores[target_idx]
        new_target = [*clusters[target_idx], customer]
        new_target_route_score = _cluster_route_proxy(instance, new_target)
        route_gain = current_route_score - new_source_route_score - new_target_route_score
        if route_gain <= min_gain:
            continue
        candidate = (route_gain, -abs(target_idx - source_idx), -target_idx)
        if best is None or candidate > (
            best.gain,
            -abs(best.target_idx - source_idx),
            -best.target_idx,
        ):
            best = _Relocation(
                target_idx=target_idx,
                gain=route_gain,
                new_source_score=new_source_score,
                new_target_score=new_target_score,
                new_source_route_score=new_source_route_score,
                new_target_route_score=new_target_route_score,
            )
    return best


def _apply_boundary_relocation(
    clusters: list[list[int]],
    loads: list[float],
    scores: list[float],
    route_scores: list[float] | None,
    source_idx: int,
    target_idx: int,
    customer: int,
    *,
    rank: dict[int, int],
    demand: float,
    new_source_score: float,
    new_target_score: float,
    new_source_route_score: float | None,
    new_target_route_score: float | None,
) -> None:
    clusters[source_idx] = [c for c in clusters[source_idx] if c != customer]
    clusters[target_idx] = _with_customer_by_rank(clusters[target_idx], customer, rank)
    loads[source_idx] -= demand
    loads[target_idx] += demand
    scores[source_idx] = new_source_score
    scores[target_idx] = new_target_score
    if route_scores is not None:
        if new_source_route_score is None or new_target_route_score is None:
            raise AssertionError("hybrid relocation must update route score cache")
        route_scores[source_idx] = new_source_route_score
        route_scores[target_idx] = new_target_route_score


def _boundary_candidates(cluster: list[int], limit: int) -> tuple[int, ...]:
    if len(cluster) <= limit:
        return tuple(cluster)
    left = limit // 2
    right = limit - left
    candidates = [*cluster[:left], *cluster[-right:]]
    return tuple(dict.fromkeys(candidates))


def _with_customer_by_rank(
    cluster: list[int],
    customer: int,
    rank: dict[int, int],
) -> list[int]:
    out = [*cluster, customer]
    out.sort(key=lambda c: (rank[c], c))
    return out


def _cluster_score(
    instance: VRPInstance,
    cluster: list[int],
    *,
    score_mode: RefinementScoreMode,
) -> float:
    if score_mode in ("compactness", "hybrid"):
        return _cluster_compactness_proxy(instance, cluster)
    return _cluster_route_proxy(instance, cluster)


def _cluster_compactness_proxy(instance: VRPInstance, cluster: list[int]) -> float:
    if not cluster:
        return 0.0
    ids = np.asarray(cluster, dtype=np.int64)
    pts = instance.coords[ids]
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    sse = float(np.einsum("ij,ij->", centered, centered))
    depot_distance = float(np.linalg.norm(centroid - instance.coords[0]))
    radial = float(np.linalg.norm(pts - instance.coords[0], axis=1).mean())
    return float(np.sqrt(max(sse, 0.0)) + 0.05 * depot_distance + 0.01 * radial)


def _cluster_route_proxy(instance: VRPInstance, cluster: list[int]) -> float:
    if not cluster:
        return 0.0
    remaining = set(cluster)
    current = 0
    cost = 0.0
    while remaining:
        nxt = min(remaining, key=lambda c: (_edge_cost(instance, current, c), c))
        cost += _edge_cost(instance, current, nxt)
        remaining.remove(nxt)
        current = nxt
    return cost + _edge_cost(instance, current, 0)


def _morton_refinement_complexity(score_mode: str) -> str:
    if score_mode == "compactness":
        return "O(n log n) sort plus O(C * k * b * m) bounded local refinement"
    if score_mode == "hybrid":
        return "O(n log n) sort plus O(C * k * b * m + C * b * s * m^2) hybrid refinement"
    return "O(n log n) sort plus O(C * k * b * m^2) bounded local refinement"


def _edge_cost(instance: VRPInstance, i: int, j: int) -> float:
    if instance.cost_matrix is not None:
        return float(instance.cost_matrix[i, j])
    diff = instance.coords[i] - instance.coords[j]
    return float(np.sqrt(np.dot(diff, diff)))


def _target_cluster_count(
    instance: VRPInstance,
    *,
    max_customers: int,
    max_demand: float,
) -> int:
    by_size = ceil(instance.num_customers / max_customers)
    by_capacity = ceil(float(instance.demand[list(instance.customers)].sum()) / max_demand)
    return min(instance.num_customers, max(1, by_size, by_capacity))


def _select_farthest_seed_positions(
    instance: VRPInstance,
    distance: np.ndarray,
    n_seeds: int,
) -> list[int]:
    customers = np.asarray(instance.customers, dtype=np.int64)
    if n_seeds >= len(customers):
        return list(range(len(customers)))

    hardness = _customer_hardness(instance, distance)
    first = max(range(len(customers)), key=lambda pos: (hardness[pos], -int(customers[pos])))
    seeds = [first]
    while len(seeds) < n_seeds:
        remaining = [pos for pos in range(len(customers)) if pos not in seeds]
        next_seed = max(
            remaining,
            key=lambda pos: (
                min(float(distance[pos, seed]) for seed in seeds),
                hardness[pos],
                -int(customers[pos]),
            ),
        )
        seeds.append(next_seed)
    return seeds


def _assignment_order(instance: VRPInstance, distance: np.ndarray) -> list[int]:
    customers = np.asarray(instance.customers, dtype=np.int64)
    hardness = _customer_hardness(instance, distance)
    return sorted(range(len(customers)), key=lambda pos: (-hardness[pos], int(customers[pos])))


def _customer_hardness(instance: VRPInstance, distance: np.ndarray) -> np.ndarray:
    customers = np.asarray(instance.customers, dtype=np.int64)
    demand_ratio = instance.demand[customers] / instance.capacity
    depot_distance = np.linalg.norm(instance.coords[customers] - instance.coords[0], axis=1)
    max_depot_distance = max(float(depot_distance.max(initial=0.0)), 1e-12)
    depot_distance = depot_distance / max_depot_distance
    window_pressure = np.zeros(instance.num_customers, dtype=np.float64)
    if instance.tw_early is not None and instance.tw_late is not None:
        width = np.maximum(instance.tw_late[customers] - instance.tw_early[customers], 0.0)
        width = np.clip(width / time_horizon(instance), 0.0, 1.0)
        window_pressure = 1.0 - width
    return demand_ratio + 0.25 * depot_distance + 0.25 * distance.mean(axis=1) + window_pressure


def _cluster_assignment_score(
    distance: np.ndarray,
    customer_pos: int,
    cluster: list[int],
    *,
    load: float,
    max_customers: int,
    max_demand: float,
) -> float:
    member_positions = [customer - 1 for customer in cluster]
    compatibility_cost = float(distance[customer_pos, member_positions].mean())
    size_balance = len(cluster) / max_customers
    load_balance = load / max_demand
    return compatibility_cost + 0.05 * size_balance + 0.05 * load_balance
