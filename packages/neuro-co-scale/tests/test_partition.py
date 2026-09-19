import numpy as np
import pytest

from neuro_co.scale.compose import compose_nearest_neighbor
from neuro_co.scale.metrics import solution_cost
from neuro_co.scale.partition import (
    capacity_aware_sweep_partition,
    grid_partition,
    morton_order,
    morton_partition,
    morton_refined_partition,
)
from neuro_co.scale.types import VRPInstance


def _instance() -> VRPInstance:
    coords = np.array(
        [
            [0.5, 0.5],
            [0.9, 0.5],
            [0.8, 0.8],
            [0.5, 0.9],
            [0.2, 0.8],
            [0.1, 0.5],
            [0.2, 0.2],
            [0.5, 0.1],
            [0.8, 0.2],
        ],
        dtype=float,
    )
    demand = np.array([0, 2, 2, 2, 2, 2, 2, 2, 2], dtype=float)
    return VRPInstance(coords=coords, demand=demand, capacity=6.0)


def test_capacity_aware_sweep_covers_each_customer_once() -> None:
    inst = _instance()
    part = capacity_aware_sweep_partition(inst, max_customers=3, capacity_fraction=1.0)

    assert sorted(part.flattened()) == list(inst.customers)
    assert all(len(cluster) <= 3 for cluster in part.clusters)
    assert all(inst.demand[list(cluster)].sum() <= inst.capacity for cluster in part.clusters)


def test_grid_partition_covers_each_customer_once() -> None:
    inst = _instance()
    part = grid_partition(inst, max_customers=2, grid_shape=(2, 2))

    assert sorted(part.flattened()) == list(inst.customers)
    assert all(len(cluster) <= 2 for cluster in part.clusters)


def test_morton_order_is_deterministic_and_covers_customers() -> None:
    inst = _instance()
    first = morton_order(inst, bits=8)
    second = morton_order(inst, bits=8)

    assert first == second
    assert sorted(first) == list(inst.customers)


def test_morton_partition_respects_size_and_capacity() -> None:
    inst = _instance()
    part = morton_partition(inst, max_customers=3, capacity_fraction=1.0, bits=8)

    assert part.method == "morton"
    assert sorted(part.flattened()) == list(inst.customers)
    assert all(len(cluster) <= 3 for cluster in part.clusters)
    assert all(inst.demand[list(cluster)].sum() <= inst.capacity for cluster in part.clusters)


def test_morton_refined_partition_moves_sparse_boundary_customer() -> None:
    coords = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [5.0, 0.0],
            [5.1, 0.0],
            [5.2, 0.0],
        ],
        dtype=float,
    )
    demand = np.array([0, 1, 1, 1, 1, 1], dtype=float)
    inst = VRPInstance(coords=coords, demand=demand, capacity=10.0)

    base = morton_partition(inst, max_customers=3, capacity_fraction=1.0, bits=8)
    refined = morton_refined_partition(
        inst,
        max_customers=3,
        capacity_fraction=1.0,
        bits=8,
        boundary_customers=2,
        neighbor_span=1,
    )

    assert refined.method == "morton_refined"
    assert refined.metadata["moves"] == 1
    assert refined.clusters == ((1, 2), (3, 4, 5))
    assert sorted(refined.flattened()) == list(inst.customers)
    assert all(len(cluster) <= 3 for cluster in refined.clusters)
    assert solution_cost(inst, compose_nearest_neighbor(inst, refined)) < solution_cost(
        inst,
        compose_nearest_neighbor(inst, base),
    )


def test_morton_refined_partition_is_deterministic_and_capacity_feasible() -> None:
    inst = _instance()
    first = morton_refined_partition(
        inst,
        max_customers=3,
        capacity_fraction=1.0,
        bits=8,
        boundary_customers=4,
        neighbor_span=1,
        max_passes=2,
    )
    second = morton_refined_partition(
        inst,
        max_customers=3,
        capacity_fraction=1.0,
        bits=8,
        boundary_customers=4,
        neighbor_span=1,
        max_passes=2,
    )

    assert first.clusters == second.clusters
    assert sorted(first.flattened()) == list(inst.customers)
    assert all(len(cluster) <= 3 for cluster in first.clusters)
    assert all(inst.demand[list(cluster)].sum() <= inst.capacity for cluster in first.clusters)


def test_morton_refined_partition_supports_route_score_mode() -> None:
    inst = _instance()
    part = morton_refined_partition(
        inst,
        max_customers=3,
        capacity_fraction=1.0,
        bits=8,
        score_mode="route",
    )

    assert part.metadata["score_mode"] == "route"
    assert sorted(part.flattened()) == list(inst.customers)


def test_morton_refined_partition_supports_hybrid_score_mode() -> None:
    inst = _instance()
    part = morton_refined_partition(
        inst,
        max_customers=3,
        capacity_fraction=1.0,
        bits=8,
        score_mode="hybrid",
        hybrid_shortlist=1,
    )

    assert part.metadata["score_mode"] == "hybrid"
    assert part.metadata["hybrid_shortlist"] == 1
    assert sorted(part.flattened()) == list(inst.customers)
    assert all(len(cluster) <= 3 for cluster in part.clusters)
    assert all(inst.demand[list(cluster)].sum() <= inst.capacity for cluster in part.clusters)


def test_morton_order_rejects_invalid_bit_depth() -> None:
    inst = _instance()

    with pytest.raises(ValueError, match="bits"):
        morton_order(inst, bits=0)


def test_morton_refined_partition_rejects_invalid_refinement_params() -> None:
    inst = _instance()

    with pytest.raises(ValueError, match="boundary_customers"):
        morton_refined_partition(inst, max_customers=3, boundary_customers=0)

    with pytest.raises(ValueError, match="score_mode"):
        morton_refined_partition(inst, max_customers=3, score_mode="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="hybrid_shortlist"):
        morton_refined_partition(inst, max_customers=3, hybrid_shortlist=0)
