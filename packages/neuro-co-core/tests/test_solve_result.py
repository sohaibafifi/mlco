"""Common outcomes preserve evidence, statuses, and JSON interoperability."""

import json

import numpy as np
import pytest

from neuro_co.core.result import SolveResult, SolveStatus, ValidationReport


def test_serialized_result_preserves_original_ids_and_validation():
    report = ValidationReport(True, 8.0, metadata={"checked": np.bool_(True)})
    result = SolveResult(
        problem="cvrp",
        method="neural",
        status=SolveStatus.FEASIBLE,
        solution={"routes": ((1, 2),), "node_ids": np.array([7, 11, 42])},
        objective=8.0,
        validation=report,
    )
    data = json.loads(result.to_json())
    assert data["schema"] == "neuro-co-solve/v1"
    assert data["solution"] == {"routes": [[1, 2]], "node_ids": [7, 11, 42]}
    assert data["feasible"] is True
    assert data["validation"]["feasible"] is True
    assert data["bound"] is None and data["gap"] is None


def test_unknown_is_distinct_from_proven_infeasibility():
    unknown = SolveResult("cvrp", "cpsat", SolveStatus.UNKNOWN, solver_status="UNKNOWN")
    infeasible = SolveResult("cvrp", "cpsat", SolveStatus.INFEASIBLE, solver_status="INFEASIBLE")
    assert unknown.feasible is None
    assert infeasible.feasible is False
    assert unknown.to_dict()["status"] != infeasible.to_dict()["status"]


@pytest.mark.parametrize(
    "field,value",
    [("objective", float("nan")), ("bound", float("inf")), ("runtime_s", -1), ("gap", -0.1)],
)
def test_result_rejects_nonfinite_or_invalid_measurements(field, value):
    with pytest.raises(ValueError):
        SolveResult("tsp", "nearest_neighbour", SolveStatus.UNKNOWN, **{field: value})


def test_infeasibility_cannot_carry_incumbent_and_validation_cannot_disagree():
    with pytest.raises(ValueError, match="incumbent"):
        SolveResult("tsp", "cpsat", SolveStatus.INFEASIBLE, objective=3)
    with pytest.raises(ValueError, match="disagrees"):
        SolveResult(
            "tsp",
            "neural",
            SolveStatus.FEASIBLE,
            validation=ValidationReport(False, None, ("missing node",)),
        )
    with pytest.raises(ValueError, match="nonfinite"):
        SolveResult("tsp", "neural", SolveStatus.UNKNOWN, metadata={"bad": float("nan")}).to_dict()


def test_status_and_gap_invariants():
    with pytest.raises(ValueError, match="feasible=False"):
        SolveResult("tsp", "cpsat", SolveStatus.INFEASIBLE, feasible=True)
    with pytest.raises(ValueError, match="unknown"):
        SolveResult("tsp", "cpsat", SolveStatus.UNKNOWN, feasible=False)
    with pytest.raises(ValueError, match="unknown"):
        SolveResult("tsp", "cpsat", SolveStatus.UNKNOWN, solution={"routes": [[1]]})
    with pytest.raises(ValueError, match="requires"):
        SolveResult("tsp", "cpsat", SolveStatus.FEASIBLE, objective=2, gap=0.2)
