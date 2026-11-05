"""Real-file adapters preserve identifiers and objective conventions."""

import json
from typing import Any

import numpy as np
import pytest

from neuro_co.problems.instances import RoutingInstance, load_instance
from neuro_co.problems.validation import validate_routes


def test_tsplib_cvrp_reindexes_explicit_ids_and_depot(tmp_path):
    path = tmp_path / "example.vrp"
    path.write_text("""NAME: example
TYPE: CVRP
DIMENSION: 3
CAPACITY: 5
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
42 3 0
7 0 0
11 0 4
DEMAND_SECTION
11 2
42 3
7 0
DEPOT_SECTION
7
-1
EOF
""")
    instance = load_instance(path)
    assert instance.node_ids == (7, 11, 42)
    assert instance.demand is not None
    assert instance.demand.tolist() == [0, 2, 3]
    assert instance.coords.tolist() == [[0, 0], [0, 4], [3, 0]]
    report = validate_routes(instance, [[1, 2]])
    assert report.feasible and report.objective == 12


def test_tsplib_rounding_is_half_up_and_explicit_matrix_is_directed(tmp_path):
    path = tmp_path / "example.tsp"
    path.write_text("""TYPE: TSP
DIMENSION: 2
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0 0
2 0.5 0
EOF
""")
    instance = load_instance(path)
    assert validate_routes(instance, [[1]]).objective == 2
    path.write_text("""TYPE: TSP
DIMENSION: 3
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: FULL_MATRIX
EDGE_WEIGHT_SECTION
0 1 9
8 0 2
3 7 0
EOF
""")
    explicit = load_instance(path)
    assert validate_routes(explicit, [[1, 2]]).objective == 6
    assert validate_routes(explicit, [[2, 1]]).objective == 24


@pytest.mark.parametrize(
    "section",
    [
        "NODE_COORD_SECTION\n1 0 0\n1 1 1\n2 2 2",
        "NODE_COORD_SECTION\n1 0 0\n2 1 1",
        "NODE_COORD_SECTION\n1 0 0\n2 1 1\n3 nan 2",
    ],
)
def test_invalid_tsplib_sections_are_rejected(tmp_path, section):
    path = tmp_path / "bad.tsp"
    path.write_text("TYPE: TSP\nDIMENSION: 3\nEDGE_WEIGHT_TYPE: EUC_2D\n" + section)
    with pytest.raises(ValueError):
        load_instance(path)


def test_json_and_npz_single_instance_contract(tmp_path):
    payload = {"problem": "cvrp", "coords": [[0, 0], [1, 0]], "demand": [0, 2], "capacity": 1}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    instance = load_instance(path)
    assert instance.demand is not None and instance.capacity is not None
    assert instance.demand[1] > instance.capacity  # mathematically valid, infeasible input
    assert not validate_routes(instance, [[1]]).feasible
    with pytest.raises(ValueError, match="differs"):
        load_instance(path, problem="tsp")
    path.write_text(json.dumps({**payload, "schema": "unknown"}))
    with pytest.raises(ValueError, match="schema"):
        load_instance(path)
    npz = tmp_path / "input.npz"
    np.savez(npz, problem="tsp", coords=np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert load_instance(npz).problem == "tsp"
    np.savez(npz, problem="tsp", coords=np.zeros((1, 3, 2)))
    with pytest.raises(ValueError, match="shape"):
        load_instance(npz)
    np.savez(npz, problem="tsp", coords=np.array([object()], dtype=object))
    with pytest.raises(ValueError, match="object-bearing"):
        load_instance(npz)


def test_route_checker_catches_coverage_invalid_indices_and_capacity():
    instance = RoutingInstance(
        "cvrp", np.array([[0, 0], [1, 0], [2, 0]]), demand=np.array([0, 2, 2]), capacity=3
    )
    for routes in ([[1, 1]], [[0, 1, 2]], [[True, 2]], [[1, 3]], [[1.0, 2]], [[1, 2]]):
        assert not validate_routes(instance, routes).feasible
    assert validate_routes(instance, [[1], [2]]).feasible
    large_distances = RoutingInstance(
        "cvrp", np.array([[0, 0], [1e12, 0]]), demand=np.array([0, 2]), capacity=1
    )
    assert not validate_routes(large_distances, [[1]]).feasible
    assert validate_routes(instance, np.array([[1], [2]])).feasible
    assert not validate_routes(instance, [[1, [2]]]).feasible
    assert not validate_routes(instance, np.array(1)).feasible


def test_route_checker_waiting_service_and_depot_return():
    options: dict[str, Any] = dict(
        problem="cvrptw",
        coords=np.array([[0, 0], [1, 0], [2, 0]]),
        demand=np.array([0, 1, 1]),
        capacity=2,
        service_time=np.array([0, 2, 0]),
    )
    instance = RoutingInstance(**options, time_windows=np.array([[0, 20], [5, 8], [0, 10]]))
    assert validate_routes(instance, [[1, 2]]).feasible
    assert validate_routes(instance, [[1, 2]]).objective == 4
    instance = RoutingInstance(**options, time_windows=np.array([[0, 9], [5, 8], [0, 10]]))
    report = validate_routes(instance, [[1, 2]])
    assert not report.feasible
    assert any("node 0" in violation for violation in report.violations)
