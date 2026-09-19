"""Tests for privacy-aware AET provenance records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuro_co.aet import provenance


def _clean_snapshot(_excluded: object = ()) -> tuple[str, bool, str | None]:
    return "a" * 40, False, None


def test_record_run_writes_checksums_without_hostname(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provenance, "_git_snapshot", _clean_snapshot)
    dataset = tmp_path / "instances.bin"
    dataset.write_bytes(b"instances")
    lock = tmp_path / "uv.lock"
    lock.write_text("locked")
    out = tmp_path / "run"

    meta = provenance.record_run(
        out,
        run_id="smoke-001",
        host_id="win-a4500-01",
        seeds={"model": 1, "instances": 2},
        config={"backend": "wall_meter", "pue": 1.0},
        artifacts={"dataset": dataset},
        lockfiles={"uv": lock},
        runtime_controls={"threads": 1},
        measurement={"scope": "wall_system"},
    )

    payload = json.loads((out / "provenance.json").read_text())
    assert payload["schema_version"] == "aet-provenance/v1"
    assert payload["run_id"] == "smoke-001"
    assert payload["host_id"] == "win-a4500-01"
    assert payload["hostname"] is None
    assert payload["python_executable"] == Path(provenance.sys.executable).name
    assert payload["artifacts"]["dataset"]["sha256"] == provenance._sha256_file(dataset)
    assert payload["artifacts"]["dataset"]["path"] == "instances.bin"
    assert payload["lockfiles"]["uv"]["sha256"] == provenance._sha256_file(lock)
    assert payload["config_sha256"] == meta.config_sha256
    assert "environment" not in payload
    assert str(tmp_path) not in json.dumps(payload)


def test_record_run_refuses_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance, "_git_snapshot", _clean_snapshot)
    provenance.record_run(tmp_path, run_id="first")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        provenance.record_run(tmp_path, run_id="second")


def test_record_run_rejects_dirty_confirmatory_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        provenance,
        "_git_snapshot",
        lambda _excluded=(): ("b" * 40, True, "c" * 64),
    )
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        provenance.record_run(tmp_path, allow_dirty=False)


def test_record_run_rejects_missing_git_for_confirmatory_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        provenance,
        "_git_snapshot",
        lambda _excluded=(): (None, False, None),
    )
    with pytest.raises(RuntimeError, match="available Git repository"):
        provenance.record_run(tmp_path, allow_dirty=False)


def test_environment_hash_does_not_depend_on_environment_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    libraries = {"torch": "1.0"}
    before = provenance._environment_hash(libraries)
    monkeypatch.setenv("AET_PRIVATE_VALUE", "first")
    middle = provenance._environment_hash(libraries)
    monkeypatch.setenv("AET_PRIVATE_VALUE", "second")
    after = provenance._environment_hash(libraries)
    assert before == middle == after


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"status": "unknown"}, "status must be one of"),
        ({"status": "failed"}, "requires failure_reason"),
        ({"status": "completed", "failure_reason": "bad"}, "cannot carry"),
        ({"started_at": "2026-08-21T10:00:00", "ended_at": "2026-08-21T11:00:00"}, "timezone"),
        (
            {
                "started_at": "2026-08-21T11:00:00+00:00",
                "ended_at": "2026-08-21T10:00:00+00:00",
            },
            "must not precede",
        ),
    ],
)
def test_record_run_rejects_ambiguous_status_or_timestamps(
    kwargs: dict[str, object],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance, "_git_snapshot", _clean_snapshot)
    with pytest.raises(ValueError, match=message):
        provenance.record_run(**kwargs)  # type: ignore[arg-type]
