from __future__ import annotations

from typing import Any

import numpy as np
import pytest

adapter = pytest.importorskip("neuro_co.problems.cvrp.pyvrp")


def _instance() -> tuple[np.ndarray, np.ndarray, float]:
    coords = np.asarray([[0.0, 0.0], [0.3, 0.4], [1.0, 0.0]])
    demands = np.asarray([0.0, 0.25, 0.5])
    return coords, demands, 0.75


def test_integer_distance_matrix_scales_rounded_euclidean_distances() -> None:
    coords, _demands, _capacity = _instance()

    matrix = adapter.integer_distance_matrix(coords, scaling_factor=100)

    np.testing.assert_array_equal(
        matrix,
        np.asarray(
            [
                [0, 50, 100],
                [50, 0, 81],
                [100, 81, 0],
            ],
            dtype=np.int64,
        ),
    )
    assert matrix.flags.c_contiguous


def test_build_problem_data_preserves_indices_and_scales_loads() -> None:
    coords, demands, capacity = _instance()

    data = adapter.build_problem_data(coords, demands, capacity, scaling_factor=100)

    assert data.num_depots == 1
    assert data.num_clients == 2
    assert data.num_vehicles == 2
    assert data.location(0).name == "depot"
    assert data.location(1).name == "client_1"
    assert data.location(2).name == "client_2"
    assert data.location(1).delivery == [25]
    assert data.location(2).delivery == [50]
    assert data.vehicle_type(0).capacity == [75]
    np.testing.assert_array_equal(
        data.distance_matrix(0), adapter.integer_distance_matrix(coords, 100)
    )


@pytest.mark.parametrize(
    ("coords", "demands", "capacity", "message"),
    [
        (np.zeros((2, 3)), np.zeros(2), 1.0, "coords must have shape"),
        (np.zeros((2, 2)), np.zeros(3), 1.0, "demands must have shape"),
        (np.zeros((2, 2)), np.asarray([1.0, 0.0]), 1.0, "depot demand must be zero"),
        (np.zeros((2, 2)), np.asarray([0.0, -1.0]), 1.0, "must be nonnegative"),
        (np.zeros((2, 2)), np.asarray([0.0, 2.0]), 1.0, "must not exceed"),
        (np.zeros((2, 2)), np.zeros(2), 0.0, "finite and positive"),
    ],
)
def test_build_problem_data_rejects_invalid_instances(
    coords: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        adapter.build_problem_data(coords, demands, capacity)


def test_solution_routes_drops_no_customer_and_contains_no_depot() -> None:
    class Route:
        def __init__(self, visits: list[int]) -> None:
            self._visits = visits

        def visits(self) -> list[int]:
            return self._visits

    class Solution:
        def routes(self) -> list[Route]:
            return [Route([2, 1]), Route([3])]

    assert adapter.solution_routes(Solution()) == ((2, 1), (3,))


def test_solution_routes_rejects_explicit_depot_marker() -> None:
    class Route:
        def visits(self) -> list[int]:
            return [1, 0, 2]

    class Solution:
        def routes(self) -> list[Route]:
            return [Route()]

    with pytest.raises(ValueError, match="depot index 0"):
        adapter.solution_routes(Solution())


def test_solve_corpus_is_sequential_and_uses_explicit_per_instance_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coords, demands, capacity = _instance()
    calls: list[dict[str, Any]] = []

    class Route:
        def visits(self) -> list[int]:
            return [1, 2]

    class Solution:
        def routes(self) -> list[Route]:
            return [Route()]

        def is_complete(self) -> bool:
            return True

    class Result:
        best = Solution()

        def __init__(self, cost: int) -> None:
            self._cost = cost

        def is_feasible(self) -> bool:
            return True

        def cost(self) -> int:
            return self._cost

    def fake_solve(
        data: Any,
        stop: Any,
        *,
        seed: int,
        collect_stats: bool,
        display: bool,
    ) -> Result:
        calls.append(
            {
                "num_clients": data.num_clients,
                "max_iterations": stop._max_iters,
                "seed": seed,
                "collect_stats": collect_stats,
                "display": display,
            }
        )
        return Result(1_250_000 + seed)

    monkeypatch.setattr(adapter, "_solve", fake_solve)
    results = adapter.solve_corpus_sequential(
        np.stack([coords, coords]),
        np.stack([demands, demands]),
        capacity,
        seed=7,
        max_iterations=23,
        collect_stats=False,
    )

    assert calls == [
        {
            "num_clients": 2,
            "max_iterations": 23,
            "seed": 7,
            "collect_stats": False,
            "display": False,
        },
        {
            "num_clients": 2,
            "max_iterations": 23,
            "seed": 8,
            "collect_stats": False,
            "display": False,
        },
    ]
    assert [result.routes for result in results] == [((1, 2),), ((1, 2),)]
    assert [result.integer_cost for result in results] == [1_250_007, 1_250_008]
    assert [result.cost for result in results] == pytest.approx([1.250007, 1.250008])
    assert [result.seed for result in results] == [7, 8]
    assert all(result.limit_kind == "iterations" for result in results)
    assert all(result.max_iterations == 23 for result in results)
    assert all(result.max_runtime_s is None for result in results)


def test_solve_corpus_time_limit_is_fresh_per_instance_and_explicit_in_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coords, demands, capacity = _instance()
    stops: list[Any] = []

    class Route:
        def visits(self) -> list[int]:
            return [1, 2]

    class Solution:
        def routes(self) -> list[Route]:
            return [Route()]

        def is_complete(self) -> bool:
            return True

    class Result:
        best = Solution()

        def is_feasible(self) -> bool:
            return True

        def cost(self) -> int:
            return 1_000_000

    def fake_solve(
        data: Any,
        stop: Any,
        *,
        seed: int,
        collect_stats: bool,
        display: bool,
    ) -> Result:
        stops.append(stop)
        return Result()

    monkeypatch.setattr(adapter, "_solve", fake_solve)
    results = adapter.solve_corpus_sequential(
        np.stack([coords, coords]),
        np.stack([demands, demands]),
        capacity,
        seed=11,
        max_runtime_s=0.25,
    )

    assert len(stops) == 2
    assert stops[0] is not stops[1]
    assert [stop._max_runtime for stop in stops] == [0.25, 0.25]
    assert [result.seed for result in results] == [11, 12]
    assert all(result.limit_kind == "time" for result in results)
    assert all(result.max_iterations is None for result in results)
    assert all(result.max_runtime_s == 0.25 for result in results)


def test_solve_corpus_runs_real_pyvrp_and_returns_depot_free_routes() -> None:
    coords, demands, capacity = _instance()

    result = adapter.solve_corpus_sequential(
        coords[None, ...],
        demands[None, ...],
        capacity,
        seed=3,
        max_iterations=20,
        scaling_factor=1_000,
    )[0]

    assert sorted(customer for route in result.routes for customer in route) == [1, 2]
    assert all(0 not in route for route in result.routes)
    assert isinstance(result.integer_cost, int)
    assert result.integer_cost > 0
    assert result.cost == result.integer_cost / 1_000


def test_solve_corpus_rejects_unbatched_or_misaligned_corpus() -> None:
    coords, demands, capacity = _instance()

    with pytest.raises(ValueError, match="corpus coords must have shape"):
        adapter.solve_corpus_sequential(
            coords,
            demands,
            capacity,
            seed=0,
            max_iterations=10,
        )
    with pytest.raises(ValueError, match="corpus demands must have shape"):
        adapter.solve_corpus_sequential(
            coords[None, ...],
            demands,
            capacity,
            seed=0,
            max_iterations=10,
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({}, ValueError, "exactly one"),
        (
            {"max_iterations": 10, "max_runtime_s": 1.0},
            ValueError,
            "exactly one",
        ),
        ({"max_runtime_s": 0.0}, ValueError, "finite and positive"),
        ({"max_runtime_s": -1.0}, ValueError, "finite and positive"),
        ({"max_runtime_s": float("nan")}, ValueError, "finite and positive"),
        ({"max_runtime_s": float("inf")}, ValueError, "finite and positive"),
        ({"max_runtime_s": True}, TypeError, "must be a number"),
        ({"max_runtime_s": object()}, TypeError, "must be a number"),
        ({"max_iterations": 0}, ValueError, "must be positive"),
        ({"max_iterations": True}, TypeError, "must be an integer"),
    ],
)
def test_solve_corpus_requires_exactly_one_positive_limit(
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
