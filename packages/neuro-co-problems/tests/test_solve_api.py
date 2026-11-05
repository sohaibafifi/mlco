"""One supplied-instance interface preserves units, solutions and solver states."""

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from neuro_co.core.factory import make_env, make_model
from neuro_co.core.result import SolveResult, SolveStatus
from neuro_co.problems import RoutingInstance, solve

api = importlib.import_module("neuro_co.problems.solve")


def _tsp(**overrides):
    values: dict[str, Any] = dict(
        problem="tsp", coords=np.array([[0.0, 0.0], [1.5, 0.0], [3.0, 0.0]]), **overrides
    )
    return RoutingInstance(**values)


def _checkpoint(tmp_path: Path, problem: str = "tsp") -> Path:
    options: dict[str, Any] = {"capacity": 5} if problem == "cvrp" else {}
    env = make_env(problem, size=4, **options)
    arch: dict[str, Any] = dict(backbone="am", hidden_dim=8, num_layers=1, num_heads=2)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = make_model(env, **arch)
    path = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "arch": arch}, path)
    (tmp_path / "config.json").write_text(
        json.dumps({"backend": "torch", "training": {"problem": problem, "size": 4}})
    )
    return path


def test_nearest_neighbour_source_units_and_rounding():
    result = solve(_tsp(distance_convention="euc_2d"), method="nearest_neighbour")
    assert result.status == SolveStatus.FEASIBLE
    assert result.solution["routes"] == [[1, 2]]
    assert result.objective == 7.0
    assert result.validation.feasible
    assert result.bound is None and result.gap is None
    assert result.to_dict()["schema"] == "neuro-co-solve/v1"


def test_explicit_matrix_and_original_ids_are_preserved():
    instance = _tsp(
        cost_matrix=[[0, 8, 1], [2, 0, 5], [7, 3, 0]],
        distance_convention="explicit",
        node_ids=(8, 2, 5),
    )
    result = solve(instance, method="nn")
    assert result.solution["routes"] == [[2, 1]]
    assert result.solution["node_ids"] == [8, 2, 5]
    assert result.objective == 6


def test_json_path_and_mapping_have_identical_solutions(tmp_path):
    data = dict(problem="tsp", coords=[[0, 0], [2, 0], [2, 3]])
    path = tmp_path / "instance.json"
    path.write_text(json.dumps(data))
    a = solve(path, method="nearest_neighbour")
    b = solve(data, method="nearest_neighbour")
    assert a.solution == b.solution and a.objective == b.objective
    with pytest.raises(ValueError, match="does not match"):
        solve(data, method="nearest_neighbour", problem="cvrp")


def test_neural_decodes_supplied_coordinates_without_sampling(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path)
    from neuro_co.problems.tsp.env import TSPEnv

    monkeypatch.setattr(TSPEnv, "reset", lambda *a, **kw: pytest.fail("sampled replacement data"))
    result = solve(_tsp(distance_convention="euc_2d"), checkpoint=checkpoint)
    assert result.status == SolveStatus.FEASIBLE
    assert sorted(result.solution["routes"][0]) == [1, 2]
    assert result.objective == 7
    assert result.validation.objective == result.objective
    assert result.metadata["checkpoint_sha256"]


def test_neural_cvrp_routes_are_complete_with_supplied_capacity(tmp_path):
    checkpoint = _checkpoint(tmp_path, "cvrp")
    instance = RoutingInstance(
        "cvrp",
        coords=[[0, 0], [10, 0], [20, 0], [20, 10]],
        demand=[0, 3, 4, 2],
        capacity=5,
    )
    result = solve(instance, checkpoint=checkpoint)
    assert result.status == SolveStatus.FEASIBLE
    assert result.validation.feasible
    assert sorted(node for route in result.solution["routes"] for node in route) == [1, 2, 3]
    assert all(
        sum(instance.demand[node] for node in route) <= 5 for route in result.solution["routes"]
    )
    assert result.objective is not None and result.objective >= 2 * 20


def test_neural_rejects_problem_and_incomplete_weights(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    data = dict(problem="cvrp", coords=[[0, 0], [1, 0], [2, 0]], demand=[0, 1, 1], capacity=2)
    with pytest.raises(ValueError, match="Checkpoint problem"):
        solve(data, checkpoint=checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    payload["model"].pop(next(iter(payload["model"])))
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        solve(_tsp(), checkpoint=checkpoint)


def test_failed_independent_check_is_invalid_not_feasible(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path)
    monkeypatch.setattr(api, "_neural_torch", lambda *a: [[1, 1]])
    result = solve(_tsp(), checkpoint=checkpoint)
    assert result.status == SolveStatus.INVALID
    assert result.feasible is False
    assert result.objective is None
    assert result.validation.violations


def test_impossible_demand_has_infeasibility_proof():
    instance = RoutingInstance("cvrp", [[0, 0], [1, 0]], demand=[0, 3], capacity=2)
    result = solve(instance, method="nn")
    assert result.status == SolveStatus.INFEASIBLE
    assert result.feasible is False and result.solution is None
    assert result.objective is None


def test_time_window_heuristic_failure_is_unknown():
    instance = RoutingInstance(
        "cvrptw",
        [[0, 0], [10, 0]],
        demand=[0, 1],
        capacity=2,
        time_windows=[[0, 2], [0, 1]],
    )
    result = solve(instance, method="nn")
    assert result.status == SolveStatus.UNKNOWN
    assert result.feasible is None and result.objective is None


def test_time_window_heuristic_accounts_for_depot_service_on_each_route():
    instance = RoutingInstance(
        "cvrptw",
        np.array([[0, 0], [1, 0], [0, 1]]),
        demand=np.array([0, 1, 1]),
        capacity=1,
        time_windows=np.array([[0, 10], [0, 4], [0, 4]]),
        service_time=np.array([2, 0, 0]),
    )
    result = solve(instance, method="nn")
    assert result.status == SolveStatus.FEASIBLE
    assert result.validation is not None and result.validation.feasible
    assert result.solution is not None and len(result.solution["routes"]) == 2


def test_classical_unknown_status_is_preserved(monkeypatch):
    adapter = importlib.import_module("neuro_co.problems.cvrp.ortools")
    monkeypatch.setattr(
        adapter,
        "solve_instance_result",
        lambda *a, **kw: SolveResult(
            "cvrp", "ortools", SolveStatus.UNKNOWN, solver_status="ROUTING_FAIL_TIMEOUT"
        ),
    )
    instance = RoutingInstance("cvrp", [[0, 0], [1, 0]], demand=[0, 1], capacity=2)
    result = solve(instance, method="ortools")
    assert result.status == SolveStatus.UNKNOWN
    assert result.solver_status == "ROUTING_FAIL_TIMEOUT"
    assert result.feasible is None and result.validation is None


@pytest.mark.parametrize("method", ["ortools", "pyvrp"])
def test_classical_cvrp_uses_original_cost_matrix(method):
    pytest.importorskip("ortools" if method == "ortools" else "pyvrp")
    instance = RoutingInstance(
        "cvrp",
        [[0, 0], [1, 0], [2, 0]],
        demand=[0, 1, 1],
        capacity=2,
        distance_convention="explicit",
        cost_matrix=[[0, 4, 1], [2, 0, 2], [3, 1, 0]],
    )
    result = solve(instance, method=method, max_runtime=0.1)
    assert result.status in {SolveStatus.FEASIBLE, SolveStatus.OPTIMAL}
    assert result.validation.feasible
    assert result.objective in {4, 9, 10}
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("method", ["cpsat", "pyvrp"])
def test_cvrptw_classical_preserves_half_up_source_distances(method):
    pytest.importorskip("ortools" if method == "cpsat" else "pyvrp")
    instance = RoutingInstance(
        "cvrptw",
        np.array([[0, 0], [0.5, 0], [1, 0]]),
        demand=np.array([0, 1, 1]),
        capacity=2,
        time_windows=np.array([[0, 10], [0, 10], [0, 10]]),
        distance_convention="euc_2d",
    )
    result = solve(instance, method=method, max_runtime=0.1)
    assert result.status in {SolveStatus.FEASIBLE, SolveStatus.OPTIMAL}
    assert result.objective == 3
    assert result.validation is not None and result.validation.feasible
    assert "reported_objective" not in result.metadata


def _jax_checkpoint(tmp_path):
    jax = pytest.importorskip("jax")
    from neuro_co.core.env_registry import make_env as make_jax_env
    from neuro_co.core.jax_backend.am import JaxAttentionModel
    from neuro_co.core.jax_backend.pomo import JaxPOMO
    from neuro_co.core.jax_backend.train import Config, save_checkpoint

    env = make_jax_env("tsp", backend="jax", size=4)
    algo = JaxPOMO(JaxAttentionModel(in_dim=2, hidden_dim=8, num_layers=1, num_heads=2), env, 2)
    checkpoint = tmp_path / "model.npz"
    save_checkpoint(
        checkpoint,
        algo.init(jax.random.key(0)),
        jax.random.key(1),
        0,
        Config(problem="tsp", size=4, n_starts=2, hidden_dim=8, num_layers=1, num_heads=2),
    )
    return checkpoint


def test_jax_decodes_a_real_checkpoint(tmp_path):
    checkpoint = _jax_checkpoint(tmp_path)
    result = solve(_tsp(), checkpoint=checkpoint, backend="jax")
    assert result.status == SolveStatus.FEASIBLE
    assert result.objective == 6
    assert result.metadata["backend"] == "jax"


@pytest.mark.parametrize("device", ["cpu:-1", "cpu:999", "cpu:invalid"])
def test_jax_device_selection_rejects_invalid_index(tmp_path, device):
    checkpoint = _jax_checkpoint(tmp_path)
    with pytest.raises(ValueError, match=r"index must be|is unavailable"):
        solve(_tsp(), checkpoint=checkpoint, backend="jax", device=device)


def test_jax_device_selection_reports_unavailable_backend(tmp_path, monkeypatch):
    jax = pytest.importorskip("jax")
    checkpoint = _jax_checkpoint(tmp_path)
    original = jax.devices

    def devices(platform=None, *args, **kwargs):
        if platform == "gpu":
            raise RuntimeError("Unknown backend gpu")
        return original(platform, *args, **kwargs)

    monkeypatch.setattr(jax, "devices", devices)
    with pytest.raises(ValueError, match="Requested JAX device 'cuda' is unavailable"):
        solve(_tsp(), checkpoint=checkpoint, backend="jax", device="cuda")


@pytest.mark.parametrize("extension", ["json", "npz"])
def test_scheduling_files_use_common_interface_and_result(tmp_path, extension):
    pytest.importorskip("ortools")
    data = dict(
        problem="jssp",
        proc_times=[[2, 0], [0, 3]],
        ops_ma_adj=[[1, 0], [0, 1]],
        ops_job_map=[0, 0],
        ops_sequence_order=[0, 1],
    )
    path = tmp_path / f"jobs.{extension}"
    if extension == "json":
        path.write_text(json.dumps(data))
    else:
        np.savez(path, **data)
    result = solve(path, method="cpsat", max_runtime=1)
    assert result.problem == "jssp"
    assert result.status == SolveStatus.OPTIMAL
    assert result.objective == 5
    assert result.solution is not None and result.solution["schedule"]
    assert result.validation is not None and result.validation.feasible
    assert result.metadata["source"] == str(path)
    assert result.metadata["solver_seed_not_configurable"] is True
    json.dumps(result.to_dict(), allow_nan=False)


def test_op_json_numeric_scalar_is_accepted(tmp_path):
    pytest.importorskip("ortools")
    data = dict(problem="op", locs=[[0, 0], [1, 0], [2, 0]], prize=[0, 1, 2], max_length=5.0)
    path = tmp_path / "orienteering.json"
    path.write_text(json.dumps(data))
    result = solve(path, method="cpsat", max_runtime=1)
    assert result.status == SolveStatus.OPTIMAL
    assert result.sense == "max"
    assert result.objective == 3


def test_json_number_precision_is_retained_for_classical_adapters(monkeypatch):
    import neuro_co.problems as package

    captured = {}

    def adapter(values, **options):
        captured.update(values)
        return SolveResult("jssp", "cpsat", SolveStatus.UNKNOWN)

    monkeypatch.setattr(package, "get_solver", lambda *a: adapter)
    result = solve({"problem": "jssp", "proc_times": [[0.1, 0.2]], "budget": 0.3}, method="cpsat")
    assert captured["proc_times"].dtype == torch.float64
    assert captured["proc_times"][0, 0].item() == 0.1
    assert captured["budget"].dtype == torch.float64
    assert captured["budget"].item() == 0.3
    assert result.metadata["instance_sha256"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "missing"},
        {"method": "nn", "seed": -1},
        {"method": "nn", "max_runtime": float("nan")},
        {"method": "nn", "backend": "jax"},
        {"method": "nn", "checkpoint": "not-used.pt"},
    ],
)
def test_unsupported_or_invalid_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        solve(_tsp(), **kwargs)
