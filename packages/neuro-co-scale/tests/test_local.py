import numpy as np
import pytest

from neuro_co.scale.local import (
    extract_local_subproblem,
    iter_local_subproblems,
    merge_local_solutions,
)
from neuro_co.scale.types import Partition, Solution, VRPInstance


def _instance() -> VRPInstance:
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
        ],
        dtype=float,
    )
    demand = np.array([0, 2, 3, 4, 5], dtype=float)
    cost_matrix = np.arange(25, dtype=float).reshape(5, 5)
    return VRPInstance(coords=coords, demand=demand, capacity=9.0, cost_matrix=cost_matrix)


def test_extract_local_subproblem_keeps_depot_first_and_maps_ids() -> None:
    inst = _instance()
    sub = extract_local_subproblem(inst, (3, 1), cluster_index=7)

    assert sub.cluster_index == 7
    assert sub.local_to_global == (0, 3, 1)
    assert sub.global_customers == (3, 1)
    assert sub.max_customer_id == 2
    assert sub.instance.num_customers == 2
    assert sub.instance.demand.tolist() == [0.0, 4.0, 2.0]
    assert sub.instance.cost_matrix is not None
    assert sub.instance.cost_matrix.tolist() == [
        [0.0, 3.0, 1.0],
        [15.0, 18.0, 16.0],
        [5.0, 8.0, 6.0],
    ]


def test_local_subproblem_maps_solution_routes_to_global_ids() -> None:
    sub = extract_local_subproblem(_instance(), (3, 1), cluster_index=2)
    local_solution = Solution(routes=((1, 2),), metadata={"solver": "dummy"})

    global_solution = sub.to_global_solution(local_solution)

    assert global_solution.routes == ((3, 1),)
    assert global_solution.metadata["solver"] == "dummy"
    assert global_solution.metadata["cluster_index"] == 2
    assert global_solution.metadata["global_customers"] == [3, 1]


def test_iter_and_merge_local_subproblems_cover_partition() -> None:
    inst = _instance()
    part = Partition(clusters=((1, 2), (3,), (4,)), method="unit")
    subproblems = iter_local_subproblems(inst, part)
    local_solutions = tuple(
        Solution(routes=(tuple(range(1, sp.max_customer_id + 1)),)) for sp in subproblems
    )

    merged = merge_local_solutions(subproblems, local_solutions)

    assert [sp.global_customers for sp in subproblems] == [(1, 2), (3,), (4,)]
    assert merged.routes == ((1, 2), (3,), (4,))


def test_extract_local_subproblem_rejects_invalid_customers() -> None:
    inst = _instance()

    with pytest.raises(ValueError, match="empty"):
        extract_local_subproblem(inst, ())
    with pytest.raises(ValueError, match="depot"):
        extract_local_subproblem(inst, (0,))
    with pytest.raises(ValueError, match="outside"):
        extract_local_subproblem(inst, (99,))
    with pytest.raises(ValueError, match="duplicate"):
        extract_local_subproblem(inst, (1, 1))


def test_merge_local_solutions_requires_matching_lengths() -> None:
    sub = extract_local_subproblem(_instance(), (1,))

    with pytest.raises(ValueError, match="same length"):
        merge_local_solutions((sub,), ())
