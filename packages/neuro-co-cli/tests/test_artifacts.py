"""Run manifests are durable, numeric and keep identity across updates."""

import json
from pathlib import Path
from typing import Any

import pytest

from neuro_co.cli.artifacts import RUN_SCHEMA, atomic_write_json, write_run


def _write(output: Path, **overrides):
    args: dict[str, Any] = dict(
        kind="train",
        problem="tsp",
        seed=7,
        backend="torch",
        config={"size": 6},
        metrics={"training_s": 1.5},
        artifacts={"metrics": "metrics.json"},
    )
    args.update(overrides)
    return write_run(output, **args)


def test_run_identity_survives_failure_and_resume(tmp_path):
    running = _write(tmp_path, status="running", metrics={})
    failed = _write(tmp_path, status="failed", metrics={}, error="interrupted")
    complete = _write(tmp_path)
    assert complete["schema"] == RUN_SCHEMA
    assert running["run_id"] == failed["run_id"] == complete["run_id"]
    assert running["created_at"] == complete["created_at"]
    assert complete["updated_at"] >= running["updated_at"]
    assert "error" not in complete
    assert json.loads((tmp_path / "run.json").read_text()) == complete


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "1", True, [1]])
def test_invalid_metrics_preserve_existing_manifest(tmp_path, value):
    original = _write(tmp_path)
    before = (tmp_path / "run.json").read_bytes()
    with pytest.raises(ValueError, match="finite number"):
        _write(tmp_path, metrics={"bad": value})
    assert (tmp_path / "run.json").read_bytes() == before
    assert json.loads(before)["run_id"] == original["run_id"]


def test_invalid_config_does_not_create_manifest(tmp_path):
    output = tmp_path / "new"
    with pytest.raises(ValueError):
        _write(output, config={"bad": float("nan")})
    assert not output.exists()


def test_artifact_references_are_relative(tmp_path):
    result = _write(
        tmp_path / "run", artifacts={"best": tmp_path / "model.pt", "local": Path("metrics.json")}
    )
    assert result["artifacts"] == {"best": "../model.pt", "local": "metrics.json"}


def test_cannot_reassign_run_identity(tmp_path):
    _write(tmp_path)
    before = (tmp_path / "run.json").read_bytes()
    with pytest.raises(ValueError, match="Cannot change run seed"):
        _write(tmp_path, seed=8)
    assert (tmp_path / "run.json").read_bytes() == before


def test_failed_atomic_replace_keeps_old_file(tmp_path, monkeypatch):
    target = tmp_path / "file.json"
    atomic_write_json(target, {"old": 1})

    def fail(*args):
        raise OSError("full disk")

    monkeypatch.setattr("neuro_co.cli.artifacts.os.replace", fail)
    with pytest.raises(OSError, match="full disk"):
        atomic_write_json(target, {"new": 2})
    assert json.loads(target.read_text()) == {"old": 1}
    assert list(tmp_path.iterdir()) == [target]
