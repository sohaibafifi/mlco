"""Versioned run metadata shared by training and experiment stages."""

from __future__ import annotations

import json
import math
import os
import platform
import tempfile
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from numbers import Real
from pathlib import Path
from typing import Any

RUN_SCHEMA = "neuro-co-run/v1"
RUN_STATUSES = frozenset({"pending", "running", "complete", "failed"})


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Serialize first and replace atomically, preserving any prior valid file."""
    path = Path(path)
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_run(
    output: str | Path,
    *,
    kind: str,
    problem: str,
    seed: int,
    backend: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, int | float | Real | None],
    artifacts: Mapping[str, str | Path],
    status: str = "complete",
    error: str | None = None,
) -> dict[str, Any]:
    """Write ``run.json`` while retaining the identity of a resumed run.

    Metrics are finite numeric scalars or null when unavailable. Artifact paths
    are stored relative to the run directory, including references to sibling
    stages. Metadata cannot silently change a run's kind, problem, or seed.
    """
    if status not in RUN_STATUSES:
        raise ValueError(f"Unknown run status {status!r}")
    if not kind or not problem or not backend:
        raise ValueError("kind, problem, and backend must be nonempty")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    numeric: dict[str, float | int | None] = {}
    for name, value in metrics.items():
        if value is None:
            numeric[str(name)] = None
        elif isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"Metric {name!r} must be a finite number or null")
        else:
            numeric[str(name)] = int(value) if isinstance(value, int) else float(value)
    output = Path(output)
    path = output / "run.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    if path.exists():
        if not isinstance(previous, dict) or previous.get("schema") != RUN_SCHEMA:
            raise ValueError(f"Unsupported run schema in {path}")
        if not isinstance(previous.get("run_id"), str) or not previous["run_id"]:
            raise ValueError(f"Missing run identity in {path}")
        for key, value in (
            ("kind", kind),
            ("problem", problem),
            ("seed", seed),
            ("backend", backend),
        ):
            if previous.get(key) != value:
                raise ValueError(f"Cannot change run {key} when updating {path}")
    references = {}
    for name, value in artifacts.items():
        target = Path(value)
        references[str(name)] = (
            os.path.relpath(target, output) if target.is_absolute() else target.as_posix()
        )
    now = datetime.now(UTC).isoformat()
    try:
        installed_version = version("neuro-co-cli")
    except PackageNotFoundError:
        installed_version = "unknown"
    payload = {
        "schema": RUN_SCHEMA,
        "run_id": previous.get("run_id") or str(uuid.uuid4()),
        "kind": kind,
        "problem": problem,
        "seed": seed,
        "backend": backend,
        "status": status,
        "created_at": previous.get("created_at", now),
        "updated_at": now,
        "config": dict(config),
        "metrics": numeric,
        "artifacts": references,
        "provenance": previous.get("provenance")
        or {"neuro_co_cli_version": installed_version, "python": platform.python_version()},
    }
    if error is not None:
        payload["error"] = str(error)
    atomic_write_json(path, payload)
    return payload


__all__ = ["RUN_SCHEMA", "RUN_STATUSES", "atomic_write_json", "write_run"]
