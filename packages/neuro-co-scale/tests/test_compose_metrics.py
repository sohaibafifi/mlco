import numpy as np

from neuro_co.scale.compose import (
    actions_from_solution,
    compose_nearest_neighbor,
    compose_time_window_aware,
)
from neuro_co.scale.metrics import evaluate_solution
from neuro_co.scale.partition import sweep_partition
from neuro_co.scale.types import Solution, VRPInstance


def _line_instance() -> VRPInstance:
    coords = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    demand = np.array([0, 4, 4, 4], dtype=float)
    return VRPInstance(coords=coords, demand=demand, capacity=8.0)


def test_compose_repairs_capacity() -> None:
    inst = _line_instance()
    part = sweep_partition(inst, max_customers=3, max_demand=100.0)
    solution = compose_nearest_neighbor(inst, part, repair_capacity=True)
    metrics = evaluate_solution(inst, solution)

    assert metrics.feasible
    assert metrics.capacity_violations == 0
    assert metrics.num_routes == 2
    assert metrics.routes_per_customer == 2 / 3
    assert metrics.cost_per_route > 0.0
    assert metrics.cost_per_customer > 0.0
    assert actions_from_solution(solution)[0] == 0
    assert actions_from_solution(solution)[-1] == 0


def test_metrics_detects_duplicate_and_missing_customers() -> None:
    inst = _line_instance()
    bad = Solution(routes=((1, 1),))
    metrics = evaluate_solution(inst, bad)

    assert not metrics.feasible
    assert metrics.duplicate_customers == 1
    assert metrics.missing_customers == 2


def test_metrics_detects_time_window_violation() -> None:
    coords = np.array([[0, 0], [10, 0]], dtype=float)
    demand = np.array([0, 1], dtype=float)
    inst = VRPInstance(
        coords=coords,
        demand=demand,
        capacity=2.0,
        tw_early=np.array([0, 0], dtype=float),
        tw_late=np.array([100, 1], dtype=float),
        service_time=np.array([0, 0], dtype=float),
    )
    metrics = evaluate_solution(inst, Solution(routes=((1,),)))

    assert not metrics.feasible
    assert metrics.time_window_violations == 1


def test_time_window_aware_constructor_prioritizes_deadlines() -> None:
    coords = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
    demand = np.array([0, 1, 1], dtype=float)
    inst = VRPInstance(
        coords=coords,
        demand=demand,
        capacity=3.0,
        tw_early=np.array([0, 10, 0], dtype=float),
        tw_late=np.array([100, 20, 3], dtype=float),
        service_time=np.zeros(3, dtype=float),
    )
    part = sweep_partition(inst, max_customers=2, max_demand=100.0)

    nearest = compose_nearest_neighbor(inst, part, repair_capacity=True)
    aware = compose_time_window_aware(inst, part, repair_capacity=True)

    assert nearest.routes == ((1, 2),)
    assert aware.routes == ((2, 1),)
    assert evaluate_solution(inst, nearest).time_window_violations == 1
    assert evaluate_solution(inst, aware).feasible
