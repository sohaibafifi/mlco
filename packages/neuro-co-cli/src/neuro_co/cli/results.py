"""Results aggregation: walk an outputs/ tree → long-format DataFrame.

Schema (one row per metric per artifact):

| column   | type           | meaning |
|----------|----------------|---------|
| problem  | str            | "vrptw", "jssp", … (parsed from path)         |
| kind     | str            | "train" / "eval" / "baseline" / "explain" / "probe" |
| method   | str or ""      | attribution method (only `kind=='explain'`)   |
| seed     | int            | parsed from `seed<N>` in path                 |
| layer    | int (-2 = NA)  | encoder layer index (probes only); -1 = final |
| concept  | str or ""      | concept name (probes only)                    |
| metric   | str            | metric name (e.g. `energy_wh`, `val_acc`)     |
| value    | float          | metric value                                  |

`collect(root)` walks `root/<problem>/{train,eval,baseline,explain}_seed<N>[_<method>]/`
plus `train_seed<N>/probes/probes.json` and emits one row per
`(file, metric)` pair. Empty / malformed JSONs are skipped with a
warning, never crash. Re-running over the same tree is idempotent.

`neuro_co.cli.figures` reads this table to produce experiment plots.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_NA_LAYER = -2  # "not applicable", compatible with pandas IntegerArray.

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
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
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
    value: float,
    method: str = "",
    layer: int = _NA_LAYER,
    concept: str = "",
) -> dict[str, Any]:
    return {
        "problem": problem,
        "kind": kind,
        "method": method,
        "seed": int(seed),
        "layer": int(layer),
        "concept": concept,
        "metric": metric,
        "value": float(value),
    }


def _collect_energy(
    problem: str, kind: str, run_dir: Path, json_name: str, keys: Iterable[str]
) -> list[dict[str, Any]]:
    seed, _ = _parse_seed_method(run_dir.name)
    if seed is None:
        return []
    data = _load_json(run_dir / json_name)
    if data is None:
        return []
    out: list[dict[str, Any]] = []
    for k in keys:
        if k in data and isinstance(data[k], (int, float)):
            out.append(_row(problem=problem, kind=kind, seed=seed, metric=k, value=float(data[k])))
    return out


def _collect_explain(problem: str, run_dir: Path) -> list[dict[str, Any]]:
    seed, method = _parse_seed_method(run_dir.name)
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
    for metric, (section, key) in _EXPLAIN_FAITH_KEYS.items():
        block = faith.get(section)
        if isinstance(block, dict) and key in block and isinstance(block[key], (int, float)):
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
        if isinstance(value, (int, float)):
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


def _collect_probes(problem: str, train_dir: Path) -> list[dict[str, Any]]:
    """Probe results live under `<train_dir>/probes/probes.json`."""
    seed, _ = _parse_seed_method(train_dir.name)
    if seed is None:
        return []
    path = train_dir / "probes" / "probes.json"
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
        layer = int(entry.get("layer", -1))
        for m in _PROBE_METRICS:
            v = entry.get(m)
            if isinstance(v, (int, float)):
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


def collect(root: str | Path) -> Any:
    """Walk an outputs/ tree, return a long-format DataFrame.

    `root` is typically `outputs/` (cross-problem rollup) or
    `outputs/<problem>/` (per-problem). Missing pandas → friendly
    ImportError.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "neuro_co.cli.results requires pandas. Install with `uv add 'neuro-co-cli[report]'`."
        ) from exc

    root_path = Path(root)
    if not root_path.is_dir():
        return pd.DataFrame(
            columns=["problem", "kind", "method", "seed", "layer", "concept", "metric", "value"]
        )

    # Detect layout:
    #   multi-problem   outputs/<problem>/<job>_seed<N>/   (root = outputs/)
    #   single-problem  outputs/<problem>/<job>_seed<N>/   (root = outputs/<problem>/)
    #
    # A problem subdirectory takes precedence over loose run directories
    # left at the root by earlier output layouts.
    _RUN_DIR_PATTERNS = ("train_seed", "eval_seed", "baseline_seed", "explain_seed")

    def _has_run_dirs(d: Path) -> bool:
        return any(
            c.is_dir() and any(c.name.startswith(pref) for pref in _RUN_DIR_PATTERNS)
            for c in d.iterdir()
        )

    sub_problem_dirs = [c for c in root_path.iterdir() if c.is_dir() and _has_run_dirs(c)]
    if sub_problem_dirs:
        problem_dirs = {c.name: c for c in sub_problem_dirs}
    elif _has_run_dirs(root_path):
        problem_dirs = {root_path.name: root_path}
    else:
        problem_dirs = {}

    rows: list[dict[str, Any]] = []
    for problem, prob_dir in problem_dirs.items():
        if not prob_dir.is_dir():
            continue
        for run_dir in sorted(prob_dir.iterdir()):
            if not run_dir.is_dir() or (
                not run_dir.name.endswith(tuple(["_seed" + d for d in "0123456789"]))
                and "seed" not in run_dir.name
            ):
                continue
            name = run_dir.name
            if name.startswith("train_seed"):
                rows += _collect_energy(problem, "train", run_dir, "energy_train.json", _TRAIN_KEYS)
                rows += _collect_probes(problem, run_dir)
            elif name.startswith("eval_seed"):
                rows += _collect_energy(problem, "eval", run_dir, "energy_eval.json", _EVAL_KEYS)
            elif name.startswith("baseline_seed"):
                rows += _collect_energy(
                    problem, "baseline", run_dir, "energy_baseline.json", _BASELINE_KEYS
                )
            elif name.startswith("explain_seed"):
                rows += _collect_explain(problem, run_dir)

    return pd.DataFrame(
        rows,
        columns=["problem", "kind", "method", "seed", "layer", "concept", "metric", "value"],
    )


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
