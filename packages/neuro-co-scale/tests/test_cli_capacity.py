"""Protect the physical demand/capacity protocol used in cross-scale studies."""

import pytest

from neuro_co.scale.cli import _make_instances


@pytest.mark.parametrize("distribution", ["uniform", "clustered"])
@pytest.mark.parametrize("capacity", [30.0, 50.0, 500.0])
def test_requested_capacity_is_preserved(distribution, capacity):
    instances = _make_instances("cvrp", 1000, 2, 123, distribution, capacity)
    assert all(instance.capacity == capacity for instance in instances)
    assert all(instance.demand[1:].min() >= 1 for instance in instances)
    assert all(instance.demand[1:].max() <= 9 for instance in instances)


def test_explicit_legacy_capacity_preserves_historical_uniform_default():
    instances = _make_instances("cvrp", 1000, 1, 123, "uniform", None)
    assert instances[0].capacity == 30
