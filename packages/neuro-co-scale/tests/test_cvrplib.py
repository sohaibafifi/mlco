from pathlib import Path

from neuro_co.scale.datasets import load_cvrplib_instance, load_cvrplib_solution
from neuro_co.scale.metrics import evaluate_solution, solution_cost


def test_cvrplib_parsing_preserves_indexing_and_rounded_cost(tmp_path: Path) -> None:
    vrp = tmp_path / "tiny.vrp"
    vrp.write_text(
        "NAME : tiny\n"
        "TYPE : CVRP\n"
        "DIMENSION : 3\n"
        "EDGE_WEIGHT_TYPE : EUC_2D\n"
        "CAPACITY : 5\n"
        "NODE_COORD_SECTION\n"
        "1 0 0\n"
        "2 1 1\n"
        "3 2 0\n"
        "DEMAND_SECTION\n"
        "1 0\n"
        "2 2\n"
        "3 3\n"
        "DEPOT_SECTION\n"
        "1\n"
        "-1\n"
        "EOF\n"
    )
    sol = tmp_path / "tiny.sol"
    sol.write_text("Route #1: 1 2\nCost 4\n")

    instance = load_cvrplib_instance(vrp)
    solution, reported_cost = load_cvrplib_solution(sol)

    assert instance.num_customers == 2
    assert instance.capacity == 5
    assert instance.demand.tolist() == [0, 2, 3]
    assert solution.routes == ((1, 2),)
    assert reported_cost == 4
    assert solution_cost(instance, solution) == 4
    assert evaluate_solution(instance, solution).feasible
