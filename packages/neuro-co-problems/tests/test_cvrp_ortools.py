from __future__ import annotations

import importlib
from collections.abc import Callable

import numpy as np
import pytest

pytest.importorskip("ortools.constraint_solver")
adapter = importlib.import_module("neuro_co.problems.cvrp.ortools")


def _instance() -> tuple[np.ndarray, np.ndarray, float]:
    coords = np.asarray(
        [[0.0, 0.0], [0.3, 0.4], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    demands = np.asarray([0.0, 0.25, 0.5, 0.5], dtype=np.float64)
    return coords, demands, 0.75


def test_integer_distance_matrix_matches_pyvrp_scaling_contract() -> None:
    coords, _demands, _capacity = _instance()

    matrix = adapter.integer_distance_matrix(coords, scaling_factor=100)

    np.testing.assert_array_equal(
        matrix,
        np.asarray(
            [
                [0, 50, 100, 100],
                [50, 0, 81, 67],
                [100, 81, 0, 141],
                [100, 67, 141, 0],
            ],
            dtype=np.int64,
        ),
    )
    assert matrix.flags.c_contiguous


def test_solution_limited_search_is_seeded_reproducible_and_valid() -> None:
    coords, demands, capacity = _instance()
    kwargs = {
        "seed": 17,
        "solution_limit": 10,
        "scaling_factor": 1_000,
    }

    first = adapter.solve_corpus_sequential(
        coords[None, ...], demands[None, ...], capacity, **kwargs
    )[0]
    second = adapter.solve_corpus_sequential(
        coords[None, ...], demands[None, ...], capacity, **kwargs
    )[0]

    assert first == second
    assert first.seed == 17
    assert first.limit_kind == "solutions"
    assert first.solution_limit == 10
    assert first.max_runtime_s is None
    assert sorted(customer for route in first.routes for customer in route) == [1, 2, 3]
    assert all(0 not in route for route in first.routes)
    assert all(sum(demands[list(route)]) <= capacity for route in first.routes)
    matrix = adapter.integer_distance_matrix(coords, 1_000)
    route_cost = sum(
        int(matrix[a, b])
        for route in first.routes
        for a, b in zip((0, *route), (*route, 0), strict=True)
    )
    assert first.integer_cost == route_cost
    assert first.cost == route_cost / 1_000


def test_corpus_uses_seed_plus_instance_index() -> None:
    coords, demands, capacity = _instance()

    results = adapter.solve_corpus_sequential(
        np.stack([coords, coords]),
        np.stack([demands, demands]),
        capacity,
        seed=23,
        solution_limit=1,
    )

    assert [result.instance_index for result in results] == [0, 1]
    assert [result.seed for result in results] == [23, 24]
    assert all(result.routes for result in results)


def test_time_limited_search_records_explicit_limit() -> None:
    coords, demands, capacity = _instance()

    result = adapter.solve_corpus_sequential(
        coords[None, ...],
        demands[None, ...],
        capacity,
        seed=5,
        max_runtime_s=1.0,
        scaling_factor=1_000,
    )[0]

    assert result.limit_kind == "time"
    assert result.max_runtime_s == 1.0
    assert result.solution_limit is None
    assert result.routes


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({}, ValueError, "exactly one"),
        (
            {"max_runtime_s": 1.0, "solution_limit": 2},
            ValueError,
            "exactly one",
        ),
        ({"max_runtime_s": 0.0}, ValueError, "finite and positive"),
        ({"max_runtime_s": float("nan")}, ValueError, "finite and positive"),
        ({"solution_limit": 0}, ValueError, "must be positive"),
        ({"solution_limit": True}, TypeError, "must be an integer"),
    ],
)
def test_search_requires_one_positive_explicit_limit(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    coords, demands, capacity = _instance()

    with pytest.raises(error, match=message):
        adapter.solve_corpus_sequential(
            coords[None, ...],
            demands[None, ...],
            capacity,
            seed=0,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("coords_transform", "demands_transform", "message"),
    [
        (lambda value: value[0], lambda value: value, "corpus coords must have shape"),
        (lambda value: value, lambda value: value[0], "corpus demands must have shape"),
    ],
)
def test_corpus_rejects_unbatched_or_misaligned_arrays(
    coords_transform: Callable[[np.ndarray], np.ndarray],
    demands_transform: Callable[[np.ndarray], np.ndarray],
    message: str,
) -> None:
    coords, demands, capacity = _instance()
    batched_coords = coords[None, ...]
    batched_demands = demands[None, ...]

    with pytest.raises(ValueError, match=message):
        adapter.solve_corpus_sequential(
            coords_transform(batched_coords),
            demands_transform(batched_demands),
            capacity,
            seed=0,
            solution_limit=1,
        )


@pytest.mark.parametrize(
    ("demands", "capacity", "message"),
    [
        (np.asarray([1.0, 0.25, 0.5, 0.5]), 0.75, "depot demand must be zero"),
        (np.asarray([0.0, -0.25, 0.5, 0.5]), 0.75, "must be nonnegative"),
        (np.asarray([0.0, 1.0, 0.5, 0.5]), 0.75, "must not exceed"),
        (np.asarray([0.0, 0.25, 0.5, 0.5]), 0.0, "finite and positive"),
    ],
)
def test_solver_rejects_invalid_capacity_inputs(
    demands: np.ndarray,
    capacity: float,
    message: str,
) -> None:
    coords, _valid_demands, _valid_capacity = _instance()

    with pytest.raises(ValueError, match=message):
        adapter.solve_corpus_sequential(
            coords[None, ...],
            demands[None, ...],
            capacity,
            seed=0,
            solution_limit=1,
        )
