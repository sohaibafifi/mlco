import numpy as np

from neuro_co.scale.compatibility import (
    CompatibilityWeights,
    routing_compatibility_distance_matrix,
)
from neuro_co.scale.features import client_feature_matrix
from neuro_co.scale.partition import feature_aware_partition
from neuro_co.scale.types import VRPInstance


def _capacity_instance() -> VRPInstance:
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


def _time_window_instance() -> VRPInstance:
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    demand = np.array([0, 1, 1, 1, 1], dtype=float)
    return VRPInstance(
        coords=coords,
        demand=demand,
        capacity=10.0,
        tw_early=np.array([0, 0, 20, 0, 20], dtype=float),
        tw_late=np.array([100, 5, 25, 5, 25], dtype=float),
        service_time=np.zeros(5, dtype=float),
    )


def test_client_feature_matrix_has_customer_rows_and_named_columns() -> None:
    inst = _capacity_instance()
    features, names = client_feature_matrix(inst)

    assert features.shape[0] == inst.num_customers
    assert features.shape[1] == len(names)
    assert "demand_ratio" in names
    assert "depot_angle_sin" in names
    assert np.isfinite(features).all()


def test_routing_compatibility_distance_matrix_is_symmetric() -> None:
    inst = _capacity_instance()
    distance = routing_compatibility_distance_matrix(inst)

    assert distance.shape == (inst.num_customers, inst.num_customers)
    np.testing.assert_allclose(distance, distance.T)
    np.testing.assert_allclose(np.diag(distance), 0.0)


def test_feature_aware_partition_respects_size_and_capacity() -> None:
    inst = _capacity_instance()
    part = feature_aware_partition(inst, max_customers=3, capacity_fraction=1.0)

    assert part.method == "feature_aware"
    assert sorted(part.flattened()) == list(inst.customers)
    assert all(len(cluster) <= 3 for cluster in part.clusters)
    assert all(inst.demand[list(cluster)].sum() <= inst.capacity for cluster in part.clusters)


def test_feature_aware_partition_can_group_by_time_window_compatibility() -> None:
    inst = _time_window_instance()
    weights = CompatibilityWeights(
        spatial=0.0,
        depot_distance=0.0,
        depot_angle=0.0,
        demand=0.0,
        time_window=1.0,
        service_time=0.0,
    )
    part = feature_aware_partition(inst, max_customers=2, capacity_fraction=1.0, weights=weights)

    assert {frozenset(cluster) for cluster in part.clusters} == {
        frozenset({1, 3}),
        frozenset({2, 4}),
    }
