"""The solve command exports machine-readable results for supplied instances."""

import json

import pytest

from neuro_co.cli.main import main
from neuro_co.cli.solve import main_cli


def _instance(tmp_path):
    path = tmp_path / "square.json"
    path.write_text(json.dumps({"problem": "tsp", "coords": [[0, 0], [1, 0], [1, 1], [0, 1]]}))
    return path


def test_solve_file_and_stdout_agree(tmp_path, capsys):
    instance = _instance(tmp_path)
    output = tmp_path / "solution.json"
    with pytest.raises(SystemExit) as exit_code:
        main(["solve", str(instance), "--out", str(output)])
    assert exit_code.value.code == 0
    result = json.loads(output.read_text())
    assert result == json.loads(capsys.readouterr().out)
    assert result["schema"] == "neuro-co-solve/v1"
    assert result["objective"] == pytest.approx(4)
    assert result["feasible"] is True
    assert result["status"] == "feasible"
    assert result["bound"] is None


def test_solve_bundle_is_collectable_and_protected(tmp_path):
    pytest.importorskip("pandas")
    from neuro_co.cli.results import collect

    instance = _instance(tmp_path)
    output = tmp_path / "bundle"
    assert main_cli([str(instance), "--out-dir", str(output)]) == 0
    result = json.loads((output / "solution.json").read_text())
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["config"]["solve_status"] == result["status"]
    rows = collect(output)
    assert set(rows["kind"]) == {"solve"}
    assert rows.loc[rows["metric"] == "objective", "value"].item() == pytest.approx(4)
    before = (output / "solution.json").read_bytes()
    with pytest.raises(SystemExit) as exit_code:
        main_cli([str(instance), "--out-dir", str(output)])
    assert exit_code.value.code == 2
    assert (output / "solution.json").read_bytes() == before


def test_invalid_instance_does_not_create_output(tmp_path):
    instance = tmp_path / "invalid.json"
    instance.write_text('{"problem":"tsp","coords":[[0,0],[1,1]],"unknown":1}')
    output = tmp_path / "result"
    with pytest.raises(SystemExit):
        main_cli([str(instance), "--method", "unsupported", "--out-dir", str(output)])
    assert not output.exists()


def test_solve_vrplib_preserves_source_ids_and_distance_convention(tmp_path, capsys):
    instance = tmp_path / "small.vrp"
    instance.write_text(
        "NAME: small\nTYPE: CVRP\nDIMENSION: 3\nCAPACITY: 2\n"
        "EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n"
        "5 0 0\n7 1.5 0\n9 0 2\nDEMAND_SECTION\n5 0\n7 1\n9 1\n"
        "DEPOT_SECTION\n5\n-1\nEOF\n"
    )
    assert main_cli([str(instance)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["solution"]["node_ids"] == [5, 7, 9]
    assert result["metadata"]["distance_convention"] == "euc_2d"
    assert result["objective"] == 7


def test_solve_scheduling_json_returns_the_same_contract(tmp_path, capsys):
    pytest.importorskip("ortools")
    instance = tmp_path / "schedule.json"
    instance.write_text(
        json.dumps(
            {
                "problem": "fjsp",
                "proc_times": [[1.0], [1.0]],
                "ops_job_map": [0, 0],
                "ops_sequence_order": [0, 1],
            }
        )
    )
    assert main_cli([str(instance), "--method", "cpsat", "--max-runtime", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["problem"] == "fjsp"
    assert result["objective"] == pytest.approx(2)
    assert result["solver_status"] == "OPTIMAL"
    assert len(result["solution"]["schedule"]) == 2


@pytest.mark.parametrize("status,expected", [("unknown", 3), ("infeasible", 2), ("invalid", 1)])
def test_solver_status_is_preserved_in_json(tmp_path, monkeypatch, capsys, status, expected):
    import neuro_co.problems
    from neuro_co.core.result import SolveResult, SolveStatus

    instance = _instance(tmp_path)
    monkeypatch.setattr(
        neuro_co.problems,
        "solve",
        lambda *a, **k: SolveResult(
            problem="tsp",
            method="test",
            status=SolveStatus(status),
            feasible=False if status in {"infeasible", "invalid"} else None,
        ),
    )
    assert main_cli([str(instance)]) == expected
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == status
    assert result["objective"] is None
