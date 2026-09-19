import numpy as np

from neuro_co.scale.partition import sweep_partition
from neuro_co.scale.partition_metrics import evaluate_partition
from neuro_co.scale.types import Partition, VRPInstance


def _instance() -> VRPInstance:
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
        ],
        dtype=float,
    )
    demand = np.array([0, 2, 2, 2, 2], dtype=float)
    return VRPInstance(coords=coords, demand=demand, capacity=4.0)


def test_evaluate_partition_reports_balance_and_coverage() -> None:
    inst = _instance()
    part = sweep_partition(inst, max_customers=2)
    metrics = evaluate_partition(inst, part)

    assert metrics.num_clusters == 2
    assert metrics.min_cluster_size == 2
    assert metrics.max_cluster_size == 2
    assert metrics.max_load_ratio == 1.0
    assert metrics.load_lower_bound_clusters == 2
    assert metrics.cluster_count_ratio_to_load_bound == 1.0
    assert metrics.clusters_per_customer == 0.5
    assert metrics.overloaded_clusters == 0
    assert metrics.missing_customers == 0
    assert metrics.duplicate_customers == 0
    assert metrics.mean_spatial_compactness >= 0.0
    assert metrics.mean_compatibility_score > 0.0


def test_evaluate_partition_detects_overload_and_coverage_errors() -> None:
    inst = _instance()
    bad = Partition(clusters=((1, 2, 3), (3,)), method="bad")
    metrics = evaluate_partition(inst, bad)

    assert metrics.overloaded_clusters == 1
    assert metrics.duplicate_customers == 1
    assert metrics.missing_customers == 1
