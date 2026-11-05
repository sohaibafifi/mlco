"""Classical adapters preserve statuses, source units and historical contracts."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from neuro_co.core.result import SolveResult, SolveStatus
from neuro_co.problems.solver_result import cp_result


class _StatusSolver:
    def StatusName(self, status):
        return status

    def WallTime(self):
        return 0.1

    def BestObjectiveBound(self):
        return 2.0


@pytest.mark.parametrize(
    ("raw", "status", "feasible"),
    [
        ("UNKNOWN", SolveStatus.UNKNOWN, None),
        ("INFEASIBLE", SolveStatus.INFEASIBLE, False),
        ("MODEL_INVALID", SolveStatus.INVALID, None),
    ],
)
def test_cp_outcomes_do_not_conflate_timeout_and_infeasibility(raw, status, feasible):
    result = cp_result("jssp", _StatusSolver(), raw)
    assert result.status == status
    if raw != "MODEL_INVALID":
        assert result.feasible is feasible
    assert result.objective is None and result.bound is None
    assert result.solver_status == raw
    assert '"schema": "neuro-co-solve/v1"' in result.to_json()


def test_rounded_model_infeasibility_is_not_a_source_proof():
    result = cp_result(
        "op", _StatusSolver(), "INFEASIBLE", metadata={"model_matches_source": False}
    )
    assert result.status == SolveStatus.UNKNOWN
    assert result.feasible is None
    assert result.solver_status == "INFEASIBLE"


def _vrptw():
    return {
        key: torch.tensor(value)
        for key, value in {
            "locs": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            "demand": [0.0, 1.0, 1.0],
            "time_windows": [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]],
            "durations": [0.0, 0.0, 0.0],
            "vehicle_capacity": 2.0,
        }.items()
    }


def test_cpsat_vrptw_common_result_and_legacy_tuple():
    pytest.importorskip("ortools")
    from neuro_co.problems.vrptw.cpsat import solve_cvrptw_cpsat

    data = _vrptw()
    result = solve_cvrptw_cpsat(data, max_runtime=0.2, return_result=True)
    assert isinstance(result, SolveResult)
    assert result.status == SolveStatus.OPTIMAL
    assert result.objective == result.bound == 4.0
    assert result.validation is not None and result.validation.feasible
    routes, cost = solve_cvrptw_cpsat(data, max_runtime=0.2)
    assert routes and cost == -4.0


def test_cpsat_distinguishes_proven_infeasible_and_unknown():
    pytest.importorskip("ortools")
    from neuro_co.problems.vrptw.cpsat import solve_cvrptw_cpsat

    data = _vrptw()
    unknown = solve_cvrptw_cpsat(data, max_runtime=1e-9, return_result=True)
    assert unknown.status == SolveStatus.UNKNOWN and unknown.feasible is None
    data["demand"][1] = 3
    infeasible = solve_cvrptw_cpsat(data, max_runtime=0.2, return_result=True)
    assert infeasible.status == SolveStatus.INFEASIBLE
    assert infeasible.feasible is False and infeasible.solution is None


def test_jssp_schedule_is_independently_validated():
    pytest.importorskip("ortools")
    from neuro_co.problems.jssp.cpsat import solve_jssp

    data = {
        key: torch.tensor(value)
        for key, value in {
            "proc_times": [[2.0, 3.0], [0.0, 0.0]],
            "ops_ma_adj": [[1, 1], [0, 0]],
            "ops_job_map": [0, 0],
            "ops_sequence_order": [0, 1],
        }.items()
    }
    result = solve_jssp(data, max_runtime=0.2, return_result=True)
    assert result.status == SolveStatus.OPTIMAL
    assert result.objective == result.bound == 5.0
    assert result.validation is not None and result.validation.feasible
    schedule, cost = solve_jssp(data, max_runtime=0.2)
    assert len(schedule) == 2 and cost == -5.0


def test_fjsp_feasibility_mode_exports_a_witness_without_optimality_claim():
    pytest.importorskip("ortools")
    from neuro_co.problems.fjsp.cpsat import solve_fjsp

    data = {
        key: torch.tensor(value)
        for key, value in {
            "proc_times": [[2.0, 4.0], [3.0, 1.0]],
            "ops_job_map": [0, 0],
            "ops_sequence_order": [0, 1],
        }.items()
    }
    result = solve_fjsp(data, max_runtime=0.2, feasibility_only=True, return_result=True)
    assert result.status == SolveStatus.FEASIBLE and result.bound is None
    assert result.solution["schedule"]
    assert result.validation is not None and result.validation.feasible
    assert solve_fjsp(data, max_runtime=0.2, feasibility_only=True) == ([], 0.0)


def test_orienteering_retains_maximization_objective():
    pytest.importorskip("ortools")
    from neuro_co.problems.op.cpsat import solve_op

    data = {
        key: torch.tensor(value)
        for key, value in {
            "locs": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            "prize": [0.0, 1.0, 2.0],
            "max_length": 4.0,
        }.items()
    }
    result = solve_op(data, max_runtime=0.2, return_result=True)
    assert result.sense == "max" and result.objective == result.bound == 3.0
    assert result.validation is not None and result.validation.feasible
    route, cost = solve_op(data, max_runtime=0.2)
    assert route[0] == route[-1] == 0 and cost == -3.0


def test_flp_relaxation_is_a_bound_without_integer_incumbent():
    pytest.importorskip("ortools")
    from neuro_co.problems.flp.lp import solve_flp

    data = {
        "locs": torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        "to_choose": torch.tensor(1),
    }
    result = solve_flp(data, max_runtime=0.2, return_result=True)
    assert result.status == SolveStatus.OPTIMAL and result.objective == 2.0
    assert result.validation is not None and result.validation.feasible
    relaxation = solve_flp(data, max_runtime=0.2, relax_only=True, return_result=True)
    assert relaxation.solution is None and relaxation.objective is None
    assert relaxation.bound == 2.0 and relaxation.metadata["relaxation"]


@pytest.mark.parametrize("engine", ["ortools", "pyvrp"])
def test_cvrp_adapters_use_source_matrix_and_validate_routes(engine):
    pytest.importorskip(engine)
    adapter = importlib.import_module(f"neuro_co.problems.cvrp.{engine}")
    matrix = np.array([[0.0, 2.0, 5.0], [2.0, 0.0, 3.0], [5.0, 3.0, 0.0]])
    result = adapter.solve_instance_result(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        [0.0, 1.0, 1.0],
        2.0,
        cost_matrix=matrix,
        max_runtime_s=0.05,
    )
    assert result.status in {SolveStatus.FEASIBLE, SolveStatus.OPTIMAL}
    assert result.objective == 10.0
    assert result.validation is not None and result.validation.feasible
    assert result.bound is None or result.metadata["model_matches_source"]


def test_cvrp_unknown_incumbent_is_not_infeasible(monkeypatch):
    pytest.importorskip("pyvrp")
    from neuro_co.problems.cvrp import pyvrp

    class NoIncumbent:
        def is_feasible(self):
            return False

    monkeypatch.setattr(pyvrp, "_solve", lambda *args, **kwargs: NoIncumbent())
    result = pyvrp.solve_instance_result([[0.0, 0.0], [1.0, 0.0]], [0.0, 1.0], 2.0)
    assert result.status == SolveStatus.UNKNOWN and result.feasible is None


def test_pdp_validates_pickup_delivery_precedence():
    pytest.importorskip("ortools")
    from neuro_co.problems.pdp.ortools import solve_pdp

    result = solve_pdp(
        {"locs": torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])},
        max_runtime=0.02,
        return_result=True,
    )
    assert result.status in {SolveStatus.FEASIBLE, SolveStatus.OPTIMAL}
    assert result.objective == 4.0
    assert result.validation is not None and result.validation.feasible


def test_fjsp_core_layout_without_pad_mask():
    pytest.importorskip("ortools")
    from neuro_co.problems.fjsp.cpsat import solve_fjsp

    data = {
        key: torch.tensor(value)
        for key, value in {
            "proc_times": [[1.0], [1.0]],
            "ops_job_map": [0, 0],
            "ops_sequence_order": [0, 1],
        }.items()
    }
    result = solve_fjsp(data, max_runtime=0.2, return_result=True)
    assert result.status == SolveStatus.OPTIMAL and result.objective == 2.0


def test_structured_jssp_rejects_ambiguous_machine_assignment():
    pytest.importorskip("ortools")
    from neuro_co.problems.jssp.cpsat import solve_jssp

    data = {
        key: torch.tensor(value)
        for key, value in {
            "proc_times": [[1.0, 2.0], [1.0, 2.0]],
            "ops_ma_adj": [[1, 1], [1, 0]],
            "ops_job_map": [0, 0],
            "ops_sequence_order": [0, 1],
        }.items()
    }
    with pytest.raises(ValueError, match="exactly one"):
        solve_jssp(data, return_result=True)


def test_orienteering_zero_budget_allows_empty_tour():
    pytest.importorskip("ortools")
    from neuro_co.problems.op.cpsat import solve_op

    data = {
        key: torch.tensor(value)
        for key, value in {
            "locs": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            "prize": [0.0, 1.0, 2.0],
            "max_length": 0.0,
        }.items()
    }
    result = solve_op(data, max_runtime=0.2, return_result=True)
    assert result.status == SolveStatus.OPTIMAL and result.objective == result.bound == 0.0
    assert result.solution["route"] == [0, 0]


def test_pdp_requires_complete_pairs():
    pytest.importorskip("ortools")
    from neuro_co.problems.pdp.ortools import solve_pdp

    with pytest.raises(ValueError, match="pairs"):
        solve_pdp({"locs": torch.zeros(4, 2)}, return_result=True)


def test_cpsat_requires_return_to_depot_before_closing():
    pytest.importorskip("ortools")
    from neuro_co.problems.vrptw.cpsat import solve_cvrptw_cpsat

    data = _vrptw()
    data["time_windows"][0, 1] = 3
    result = solve_cvrptw_cpsat(data, max_runtime=0.2, return_result=True)
    assert result.status == SolveStatus.INFEASIBLE


def test_pdp_large_coordinate_units_do_not_introduce_a_travel_budget():
    pytest.importorskip("ortools")
    from neuro_co.problems.pdp.ortools import solve_pdp

    result = solve_pdp(
        {"locs": torch.tensor([[0.0, 0.0], [100000.0, 0.0], [200000.0, 0.0]])},
        max_runtime=0.02,
        return_result=True,
    )
    assert result.feasible is True and result.objective == 400000.0
