import numpy as np
import pytest

from neuro_co.scale.cluster_encoder import ClusterLocalEncoder
from neuro_co.scale.decode import HierarchicalPolicy
from neuro_co.scale.eval import (
    MethodEval,
    evaluate_constructor,
    evaluate_policy,
    paired_gap,
)
from neuro_co.scale.hierarchical import LinearClusterSelector
from neuro_co.scale.partition import morton_partition
from neuro_co.scale.types import VRPInstance

core_models = pytest.importorskip("neuro_co.core.models")


def _instances(n: int = 3, num_customers: int = 24) -> list[VRPInstance]:
    out = []
    for seed in range(n):
        rng = np.random.default_rng(seed)
        coords = rng.random((num_customers + 1, 2))
        demand = np.zeros(num_customers + 1)
        demand[1:] = rng.integers(1, 10, size=num_customers)
        out.append(VRPInstance(coords=coords, demand=demand, capacity=30.0))
    return out


def _policy(hidden_dim: int = 16) -> HierarchicalPolicy:
    encoder = ClusterLocalEncoder(in_dim=3, hidden_dim=hidden_dim, num_layers=1, num_heads=2)
    selector = LinearClusterSelector(hidden_dim=hidden_dim)
    pointer = core_models.PointerDecoder(hidden_dim=hidden_dim, num_heads=2)
    return HierarchicalPolicy(encoder, selector, pointer)


def _partition_fn():
    return lambda instance: morton_partition(instance, max_customers=8)


def test_evaluate_constructor_reports_feasible_costs() -> None:
    instances = _instances()
    result = evaluate_constructor(instances, _partition_fn(), name="morton+nn")

    assert result.name == "morton+nn"
    assert len(result.costs) == len(instances)
    assert result.feasible_rate == 1.0
    assert result.mean_cost > 0
    assert result.mean_routes > 0


def test_evaluate_policy_matches_instance_count_and_is_feasible() -> None:
    instances = _instances()
    result = evaluate_policy(_policy(), instances, _partition_fn(), mode="greedy")

    assert len(result.costs) == len(instances)
    assert result.feasible_rate == 1.0
    assert all(c > 0 for c in result.costs)


def test_paired_gap_is_zero_against_itself() -> None:
    instances = _instances()
    baseline = evaluate_constructor(instances, _partition_fn(), name="morton+nn")

    gap = paired_gap(baseline, baseline)

    assert gap.mean_gap_pct == pytest.approx(0.0)
    assert gap.win_rate == 0.0  # strict less-than against identical costs


def test_paired_gap_requires_matched_instances() -> None:
    short = evaluate_constructor(_instances(2), _partition_fn(), name="a")
    long = evaluate_constructor(_instances(3), _partition_fn(), name="b")

    with pytest.raises(ValueError, match="same instances"):
        paired_gap(short, long)


def test_paired_gap_direction_for_cheaper_method() -> None:
    cheaper = MethodEval("cheap", 9.0, 1.0, 3.0, costs=(8.0, 10.0, 9.0))
    pricier = MethodEval("pricey", 12.0, 1.0, 3.0, costs=(10.0, 14.0, 12.0))

    gap = paired_gap(cheaper, pricier)

    assert gap.mean_gap_pct < 0.0  # cheaper sits below the reference
    assert gap.win_rate == 1.0  # wins every paired instance
