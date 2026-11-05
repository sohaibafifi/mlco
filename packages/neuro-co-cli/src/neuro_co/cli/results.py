"""Aggregate versioned runs and historical artifacts into a long-format table.

Each scalar metric has one row. The original problem/kind/method/seed/layer/
concept/metric/value columns are followed by run_id, backend, size, status and
schema. ``run.json`` supplies modern metadata without directory conventions.
Historical metrics/config/evaluation and energy/probe/explanation sidecars are
also accepted. A manifest wins when it repeats a historical metric. Runs with
no numeric metrics retain a row with an empty metric and null value so failed
and pending work remains visible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeGuard

from .artifacts import RUN_SCHEMA, RUN_STATUSES

log = logging.getLogger(__name__)

_NA_LAYER = -2  # "not applicable", compatible with pandas IntegerArray.
_COLUMNS = [
    "problem",
    "kind",
    "method",
    "seed",
    "layer",
    "concept",
    "metric",
    "value",
    "run_id",
    "backend",
    "size",
    "status",
    "schema",
]


def _number(value: Any) -> TypeGuard[int | float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


# Metrics pulled from each JSON kind. Missing keys are skipped silently.
_TRAIN_KEYS = (
    "energy_wh",
    "co2_g_total",
    "co2_g_operational",
    "duration_s",
    "avg_power_w",
)
_EVAL_KEYS = (
    "energy_wh",
    "energy_wh_per_item",
    "co2_g_total",
    "gap_to_bks",
    "throughput_items_per_s",
    "nn_cost",
    "baseline_cost",
)
_BASELINE_KEYS = (
    "energy_wh",
    "co2_g_total",
    "avg_cost",
    "duration_s",
    "baseline_gap",
    "size",
)
# Explanation faithfulness payload lives under `faithfulness.{deletion,sufficiency,sanity_check}`.
_EXPLAIN_FAITH_KEYS = {
    "deletion.mean_flip_rate": ("deletion", "mean_flip_rate"),
    "sufficiency.mean_keep_rate": ("sufficiency", "mean_keep_rate"),
    "sanity.mean_jaccard": ("sanity_check", "mean_jaccard"),
    "sanity.chance_jaccard": ("sanity_check", "chance_jaccard"),
}
_PROBE_METRICS = ("val_acc", "val_balanced_acc", "val_f1", "val_roc_auc")

_SEED_RX = re.compile(r"seed(?P<seed>\d+)(?:_(?P<method>[A-Za-z0-9_-]+))?$")


def _load_json(path: Path) -> dict[str, Any] | None:
    """Return parsed JSON, `None` if missing or malformed.

    Missing files are silent (most run dirs only carry a subset of
    artifacts depending on which stages ran). Malformed JSON gets a
    WARNING so callers can identify corrupt files.
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data
    except (OSError, ValueError) as exc:
        log.warning("results: skipping malformed %s: %s", path, exc)
        return None


def _parse_seed_method(dir_name: str) -> tuple[int | None, str]:
    """Pull `seed<N>` + optional `_<method>` suffix out of a run-dir name."""
    m = _SEED_RX.search(dir_name)
    if m is None:
        return None, ""
    return int(m.group("seed")), (m.group("method") or "")


def _row(
    *,
    problem: str,
    kind: str,
    seed: int,
    metric: str,
    value: float | None,
    method: str = "",
    layer: int = _NA_LAYER,
    concept: str = "",
    run_id: str = "",
    backend: str = "unknown",
    size: int | None = None,
    status: str = "complete",
    schema: str = "legacy",
) -> dict[str, Any]:
    return {
        "problem": problem,
        "kind": kind,
        "method": method,
        "seed": int(seed),
        "layer": int(layer),
        "concept": concept,
        "metric": metric,
        "value": float(value) if value is not None else None,
        "run_id": run_id,
        "backend": backend,
        "size": size,
        "status": status,
        "schema": schema,
    }


def _collect_energy(
    problem: str,
    kind: str,
    run_dir: Path,
    json_name: str,
    keys: Iterable[str],
    seed_override: int | None = None,
) -> list[dict[str, Any]]:
    seed = seed_override if seed_override is not None else _parse_seed_method(run_dir.name)[0]
    if seed is None:
        return []
    data = _load_json(run_dir / json_name)
    if data is None:
        return []
    out: list[dict[str, Any]] = []
    for k in keys:
        if k in data and _number(data[k]):
            out.append(_row(problem=problem, kind=kind, seed=seed, metric=k, value=float(data[k])))
    return out


def _collect_explain(
    problem: str,
    run_dir: Path,
    seed_override: int | None = None,
    method_override: str | None = None,
) -> list[dict[str, Any]]:
    seed, method = _parse_seed_method(run_dir.name)
    if seed_override is not None:
        seed = seed_override
    if method_override is not None:
        method = method_override
    if seed is None:
        return []
    data = _load_json(run_dir / "explanation.json")
    if data is None:
        return []
    # `method` may also be recorded in the JSON itself; the dir suffix wins
    # because that's how `neuroco sweep` keys runs.
    method = method or str(data.get("method", ""))
    out: list[dict[str, Any]] = []
    faith = data.get("faithfulness", {})
    if not isinstance(faith, dict):
        return []
    for metric, (section, key) in _EXPLAIN_FAITH_KEYS.items():
        block = faith.get(section)
        if isinstance(block, dict) and key in block and _number(block[key]):
            out.append(
                _row(
                    problem=problem,
                    kind="explain",
                    seed=seed,
                    method=method,
                    metric=metric,
                    value=float(block[key]),
                )
            )
    # Top-level convenience aliases for older traces.
    legacy = {
        "deletion.mean_flip_rate": faith.get("mean_flip_rate"),
        "sufficiency.mean_keep_rate": faith.get("mean_keep_rate"),
    }
    for metric, value in legacy.items():
        if _number(value):
            already = any(r["metric"] == metric for r in out)
            if not already:
                out.append(
                    _row(
                        problem=problem,
                        kind="explain",
                        seed=seed,
                        method=method,
                        metric=metric,
                        value=float(value),
                    )
                )
    return out


def _collect_probes(
    problem: str,
    train_dir: Path,
    seed_override: int | None = None,
    artifact_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Probe results live under `<train_dir>/probes/probes.json`."""
    seed = seed_override if seed_override is not None else _parse_seed_method(train_dir.name)[0]
    if seed is None:
        return []
    path = artifact_path if artifact_path is not None else train_dir / "probes" / "probes.json"
    data = _load_json(path)
    if data is None:
        return []
    probes = data.get("probes") if isinstance(data, dict) else None
    if not isinstance(probes, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in probes:
        if not isinstance(entry, dict):
            continue
        concept = str(entry.get("concept", ""))
        try:
            layer = int(entry.get("layer", -1))
        except (TypeError, ValueError):
            continue
        for m in _PROBE_METRICS:
            v = entry.get(m)
            if _number(v):
                out.append(
                    _row(
                        problem=problem,
                        kind="probe",
                        seed=seed,
                        layer=layer,
                        concept=concept,
                        metric=m,
                        value=float(v),
                    )
                )
    return out


def _config_values(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    nested = config.get("training")
    return nested if isinstance(nested, dict) else config


def _metadata(data: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    config = _config_values(data.get("config", {}))
    size = config.get("size")
    return {
        "run_id": data.get("run_id")
        or "legacy-" + hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()[:16],
        "backend": data.get("backend", config.get("backend", "unknown")),
        "size": size if isinstance(size, int) and not isinstance(size, bool) else None,
        "status": data.get("status", "complete"),
        "schema": data.get("schema", "legacy"),
    }


def _manifest_rows(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    data = _load_json(path)
    if data is None:
        return None, []
    metrics = data.get("metrics")
    valid = (
        data.get("schema") == RUN_SCHEMA
        and isinstance(data.get("run_id"), str)
        and bool(data.get("run_id"))
        and isinstance(data.get("problem"), str)
        and isinstance(data.get("kind"), str)
        and isinstance(data.get("seed"), int)
        and not isinstance(data.get("seed"), bool)
        and data["seed"] >= 0
        and isinstance(data.get("status"), str)
        and data.get("status") in RUN_STATUSES
        and isinstance(metrics, dict)
    )
    if not valid:
        log.warning("results: skipping invalid run manifest %s", path)
        return None, []
    assert isinstance(metrics, dict)
    config = _config_values(data.get("config", {}))
    common = {
        "problem": data["problem"],
        "kind": data["kind"],
        "seed": data["seed"],
        "method": str(config.get("method", "")),
        **_metadata(data, path.parent),
    }
    rows = [
        _row(**common, metric=str(key), value=value)
        for key, value in metrics.items()
        if _number(value)
    ]
    if not rows:
        # Retain failed, running and pending runs without inventing a score.
        rows = [_row(**common, metric="", value=None)]
    return data, rows


def _standard_rows(run_dir: Path, manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    metrics = _load_json(run_dir / "metrics.json") or {}
    saved = _load_json(run_dir / "config.json") or {}
    config = _config_values(saved or metrics.get("args", {}))
    if not config and manifest:
        config = _config_values(manifest.get("config", {}))
    problem = config.get("problem") or (manifest or {}).get("problem")
    seed = config.get("seed", (manifest or {}).get("seed"))
    if not isinstance(problem, str) or not isinstance(seed, int) or isinstance(seed, bool):
        return []
    meta = _metadata(
        {
            "config": config,
            "backend": saved.get("backend", metrics.get("backend", "torch")),
            **(manifest or {}),
        },
        run_dir,
    )
    rows: list[dict[str, Any]] = []
    if metrics and not manifest:
        values = {key: metrics[key] for key in ("best_reward", "best_step") if key in metrics}
        timing = metrics.get("timing", {})
        if isinstance(timing, dict):
            values.update(timing)
        history = metrics.get("history")
        if isinstance(history, list) and history and isinstance(history[-1], dict):
            values.update({f"last.{key}": value for key, value in history[-1].items()})
        rows.extend(
            _row(problem=problem, kind="train", seed=seed, metric=str(key), value=value, **meta)
            for key, value in values.items()
            if _number(value)
        )
    if manifest is None or manifest.get("kind") == "train":
        evaluation = _load_json(run_dir / "evaluation.json") or {}
        rows.extend(
            _row(problem=problem, kind="eval", seed=seed, metric=str(key), value=value, **meta)
            for key, value in evaluation.items()
            if _number(value)
        )
    return rows


def collect(root: str | Path) -> Any:
    """Collect versioned manifests and legacy runs under exactly ``root``.

    Modern runs may use arbitrary directory names. A manifest takes precedence
    over legacy metrics for its stage; legacy energy, probes and explanation
    artifacts remain readable. Rows carry run identity and completion status.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "neuro_co.cli.results requires pandas. Install neuro-co-cli[report]."
        ) from exc
    root_path = Path(root)
    if not root_path.is_dir():
        return pd.DataFrame(columns=pd.Index(_COLUMNS))
    manifests: dict[Path, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for path in sorted(root_path.rglob("run.json")):
        data, modern_rows = _manifest_rows(path)
        if data:
            manifests[path.parent] = data
            rows.extend(modern_rows)
    candidates = set(manifests)
    for name in (
        "metrics.json",
        "evaluation.json",
        "energy_train.json",
        "energy_eval.json",
        "energy_baseline.json",
        "explanation.json",
    ):
        candidates.update(path.parent for path in root_path.rglob(name))
    candidates.update(
        path.parent.parent
        for path in root_path.rglob("probes.json")
        if path.parent.name == "probes"
    )
    for run_dir in sorted(candidates):
        manifest = manifests.get(run_dir)
        rows.extend(_standard_rows(run_dir, manifest))
        seed, _ = _parse_seed_method(run_dir.name)
        if manifest:
            problem = manifest["problem"]
            metadata = _metadata(manifest, run_dir)
        elif seed is not None:
            problem = run_dir.parent.name
            metadata = _metadata({}, run_dir)
        else:
            continue
        legacy = []
        kind = manifest["kind"] if manifest else run_dir.name.split("_seed", 1)[0]
        seed = manifest["seed"] if manifest else seed
        if kind == "train":
            legacy.extend(
                _collect_energy(problem, "train", run_dir, "energy_train.json", _TRAIN_KEYS, seed)
            )
            legacy.extend(_collect_probes(problem, run_dir, seed))
        elif kind == "eval":
            legacy.extend(
                _collect_energy(problem, "eval", run_dir, "energy_eval.json", _EVAL_KEYS, seed)
            )
        elif kind == "baseline":
            legacy.extend(
                _collect_energy(
                    problem, "baseline", run_dir, "energy_baseline.json", _BASELINE_KEYS, seed
                )
            )
        elif kind == "probe" and manifest:
            artifacts = manifest.get("artifacts", {})
            reference = (
                artifacts.get("probes", "probes.json")
                if isinstance(artifacts, dict)
                else "probes.json"
            )
            if isinstance(reference, str):
                probe_path = (run_dir / reference).resolve()
                if probe_path.is_relative_to(root_path.resolve()):
                    legacy.extend(_collect_probes(problem, run_dir, seed, probe_path))
                else:
                    log.warning(
                        "results: skipping probe reference outside collection root: %s", probe_path
                    )
        elif kind == "explain":
            method = (
                str(_config_values(manifest.get("config", {})).get("method", ""))
                if manifest
                else None
            )
            legacy.extend(_collect_explain(problem, run_dir, seed, method))
        for row in legacy:
            row.update(metadata)
            rows.append(row)
    populated = {(row["run_id"], row["kind"]) for row in rows if row["metric"]}
    rows = [row for row in rows if row["metric"] or (row["run_id"], row["kind"]) not in populated]
    frame = pd.DataFrame(rows, columns=pd.Index(_COLUMNS))
    if not frame.empty:
        # Prefer modern rows when a legacy sidecar repeats the same metric.
        frame = frame.drop_duplicates(
            subset=["run_id", "kind", "method", "layer", "concept", "metric"], keep="first"
        ).reset_index(drop=True)
    return frame


def write_table(df: Any, out: Path) -> None:
    """Write the DataFrame to parquet (default) or csv (by extension)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".csv":
        df.to_csv(out, index=False)
    else:
        df.to_parquet(out, index=False)


def main_cli(argv: list[str] | None = None) -> int:
    """Entry called by the `neuroco results` subcommand."""
    import argparse

    parser = argparse.ArgumentParser(prog="neuroco results")
    parser.add_argument("root", type=Path, help="Path to outputs/ or outputs/<problem>/.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results.parquet"),
        help="Output file. `.parquet` (default) or `.csv` by extension.",
    )
    ns = parser.parse_args(argv)

    df = collect(ns.root)
    write_table(df, ns.out)
    print(f"[results] {len(df)} rows -> {ns.out}", flush=True)
    if not df.empty:
        summary = df.groupby(["problem", "kind"])["metric"].nunique().reset_index(name="n_metrics")
        print(summary.to_string(index=False), flush=True)
    return 0
