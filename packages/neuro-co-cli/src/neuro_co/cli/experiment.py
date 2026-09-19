"""Experiment-recipe DSL: declarative YAML → reproducible artifact bundle.

An *experiment* is a YAML file under `recipes/<name>.yaml`
describing the problems, seeds, stages, and method sweeps that
make up one experiment. `neuroco experiment <yaml>` reads
the recipe, expands it into the equivalent `neuroco sweep` /
`neuroco results` / `neuroco figures` calls, and lands every
artifact under `experiments/<name>/` (results.parquet + figs/ +
manifest.json + per-run dirs already under `outputs/<problem>/`).

Path convention: recipes (input) and experiment bundles (output)
live in separate top-level dirs so the inputs / outputs don't
intermix. `recipes/foo.yaml` -> `experiments/foo/`.

`neuroco budget <yaml>` reports expected wall-clock without
running anything. Past `energy_*.json` files under `outputs/` are
re-used when they match a planned signature; otherwise rough
per-stage constants provide an estimate.

YAML schema (minimal):

```yaml
name: baseline_v1
description: |
  Five problems x 3 seeds x 4 attribution methods.
parallel: 4                              # ProcessPoolExecutor workers
problems:
  vrptw:
    seeds: [0, 1, 2]
    train:
      overrides: ["trainer.max_epochs=5"]
    baseline:
      overrides: ["baseline.num_problems=20"]
    eval:
      overrides: []
    explain:
      methods: [gradient, ig, contrastive, deeplift]
      overrides: ["xai.run_sufficiency=true"]
    probe:
      args: ["--num-instances", "16", "--epochs", "200"]
  jssp:
    seeds: [0, 1, 2]
    train:
      overrides: ["trainer.max_epochs=5"]
    explain:
      methods: [gradient]
      overrides: []
```

The generic executor currently implements only `train` and `probe`.
Recipes containing `baseline`, `eval`, or `explain` are rejected before any
output directory is created. Dedicated runners must be added before those
stages can be executed safely.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_STAGE_ORDER: tuple[str, ...] = ("train", "baseline", "eval", "explain", "probe")
_IMPLEMENTED_EXECUTION_STAGES: frozenset[str] = frozenset({"train", "probe"})

# Rough per-stage wall-clock fallbacks (seconds) when no prior run
# matches the planned signature. Used only by `estimate-budget`.
# These are intentionally conservative: overshooting the estimate
# is better than under-budgeting and timing out a SLURM job.
_DEFAULT_STAGE_SECONDS: dict[str, float] = {
    "train": 60.0,
    "baseline": 30.0,
    "eval": 30.0,
    "explain": 20.0,
    "probe": 15.0,
}


# ---------------------------------------------------------------------------
# Parsing.
# ---------------------------------------------------------------------------


@dataclass
class ExperimentRecipe:
    name: str
    description: str
    parallel: int
    problems: dict[str, dict[str, Any]]
    raw: dict[str, Any]  # original YAML for the manifest


def load_recipe(path: Path) -> ExperimentRecipe:
    """Parse + validate an experiment YAML. Missing pyyaml -> friendly error."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "neuroco experiment needs PyYAML. Install via `uv sync --all-extras`."
        ) from exc
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise SystemExit(f"experiment YAML must be a mapping, got {type(raw).__name__}")
    name = str(raw.get("name") or Path(path).stem)
    description = str(raw.get("description") or "")
    parallel = int(raw.get("parallel", 1))
    problems = raw.get("problems") or {}
    if not isinstance(problems, dict) or not problems:
        raise SystemExit("experiment YAML must declare at least one problem under `problems:`.")
    for prob, cfg in problems.items():
        if not isinstance(cfg, dict):
            raise SystemExit(f"problem {prob!r} entry must be a mapping")
        seeds = cfg.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise SystemExit(f"problem {prob!r} must declare a non-empty `seeds:` list")
        for s in seeds:
            int(s)  # raise if non-numeric
        for stage in cfg:
            if stage in {"seeds"}:
                continue
            if stage not in _STAGE_ORDER:
                raise SystemExit(
                    f"problem {prob!r}: unknown stage {stage!r} (allowed: {_STAGE_ORDER})"
                )
    return ExperimentRecipe(
        name=name,
        description=description,
        parallel=parallel,
        problems=problems,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Execution.
# ---------------------------------------------------------------------------


def _seeds(cfg: dict[str, Any]) -> list[int]:
    return [int(s) for s in cfg["seeds"]]


def _stage(cfg: dict[str, Any], stage: str) -> dict[str, Any] | None:
    block = cfg.get(stage)
    if block is None:
        return None
    if not isinstance(block, dict):
        raise SystemExit(f"stage {stage!r} block must be a mapping")
    return block


def _run_stage_sweep(
    problem: str,
    target: str,
    stage_cfg: dict[str, Any],
    seeds: list[int],
    parallel: int,
) -> None:
    """Run the core-native training sweep over seeds for a `train` stage."""
    if target != "train":
        raise SystemExit(
            f"experiment stage {target!r} has no dedicated runner; refusing to call "
            "the training runner for a non-training stage"
        )

    from neuro_co.cli.runners import TrainArgs, train_run

    cfg = {str(k): v for k, v in stage_cfg.items()}
    algo = str(cfg.get("algo", "reinforce"))
    size = int(cfg.get("size", 50))
    epochs = int(cfg.get("epochs", 10))
    for seed in seeds:
        out_dir = Path("outputs") / problem / f"{target}_seed{seed}"
        train_run(
            TrainArgs(
                problem=problem,
                algo=algo,
                size=size,
                epochs=epochs,
                seed=seed,
                out_dir=str(out_dir),
            )
        )


def _validate_execution_plan(recipe: ExperimentRecipe) -> None:
    """Reject unsupported stages before creating outputs or starting any run."""
    unsupported = sorted(
        {
            stage
            for problem_cfg in recipe.problems.values()
            for stage in _STAGE_ORDER
            if stage in problem_cfg and stage not in _IMPLEMENTED_EXECUTION_STAGES
        }
    )
    if unsupported:
        rendered = ", ".join(unsupported)
        raise SystemExit(
            "experiment recipe requests stages without dedicated runners: "
            f"{rendered}. Refusing execution before any training or output write."
        )


def _run_stage_probe(problem: str, stage_cfg: dict[str, Any], seeds: list[int]) -> None:
    """Loop seeds and invoke `neuroco probe`."""
    extra_args = list(stage_cfg.get("args") or [])
    for seed in seeds:
        run_dir = Path("outputs") / problem / f"train_seed{seed}"
        if not run_dir.is_dir():
            log.warning("[experiment] probe: missing %s, skip", run_dir)
            continue
        cmd = ["uv", "run", "neuroco", "probe", str(run_dir), *extra_args]
        log.info("[experiment] %s", " ".join(cmd))
        subprocess.run(cmd, check=False)


def run_experiment(yaml_path: Path) -> int:
    """Expand a YAML recipe into `neuroco` calls + aggregate artifacts."""
    from neuro_co.cli.figures import render_all
    from neuro_co.cli.results import collect, write_table

    recipe = load_recipe(yaml_path)
    _validate_execution_plan(recipe)
    exp_dir = Path("experiments") / recipe.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = exp_dir / "figs"

    manifest = {
        "name": recipe.name,
        "description": recipe.description,
        "yaml_path": str(yaml_path),
        "started_at": _now_iso(),
        "git_sha": _git_sha(),
        "recipe": recipe.raw,
    }
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for problem, pcfg in recipe.problems.items():
        seeds = _seeds(pcfg)
        for stage in _STAGE_ORDER:
            scfg = _stage(pcfg, stage)
            if scfg is None:
                continue
            log.info("[experiment] %s/%s seeds=%s", problem, stage, seeds)
            if stage == "probe":
                _run_stage_probe(problem, scfg, seeds)
            else:
                _run_stage_sweep(
                    problem=problem,
                    target=stage,
                    stage_cfg=scfg,
                    seeds=seeds,
                    parallel=recipe.parallel,
                )

    df = collect("outputs/")
    results_path = exp_dir / "results.parquet"
    write_table(df, results_path)

    figs = render_all(df, figs_dir)
    manifest["finished_at"] = _now_iso()
    manifest["n_rows"] = len(df)
    manifest["figures"] = [str(p.relative_to(exp_dir)) for p in figs]
    manifest["results"] = str(results_path.relative_to(exp_dir))
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(
        f"[experiment] {recipe.name} -> {exp_dir} (rows={len(df)}, figures={len(figs)})",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# Budget audit.
# ---------------------------------------------------------------------------


def _energy_files_for(problem: str, stage: str, seed: int) -> list[Path]:
    """Find existing `energy_*.json` matching (problem, stage, seed)."""
    root = Path("outputs") / problem
    if not root.is_dir():
        return []
    if stage == "train":
        pat, fname = f"train_seed{seed}*", "energy_train.json"
    elif stage == "eval":
        pat, fname = f"eval_seed{seed}*", "energy_eval.json"
    elif stage == "baseline":
        pat, fname = f"baseline_seed{seed}*", "energy_baseline.json"
    elif stage == "explain":
        pat, fname = f"explain_seed{seed}*", "explanation.json"
    elif stage == "probe":
        pat, fname = f"train_seed{seed}*", "probes/probes.json"
    else:
        return []
    return [p / fname for p in root.glob(pat) if (p / fname).is_file()]


def _read_duration(path: Path) -> float | None:
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    v = d.get("duration_s") or d.get("per_problem_runtime_s")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list) and v:
        return float(sum(v))
    return None


def estimate_budget(yaml_path: Path) -> dict[str, Any]:
    """Estimate per-(problem, stage) wall-clock; return a summary dict."""
    recipe = load_recipe(yaml_path)
    rows: list[dict[str, Any]] = []
    for problem, pcfg in recipe.problems.items():
        seeds = _seeds(pcfg)
        for stage in _STAGE_ORDER:
            scfg = _stage(pcfg, stage)
            if scfg is None:
                continue
            n_methods = max(1, len(scfg.get("methods") or [])) if stage == "explain" else 1
            # Per-seed duration estimate.
            durations: list[float] = []
            for seed in seeds:
                files = _energy_files_for(problem, stage, seed)
                for f in files:
                    d = _read_duration(f)
                    if d is not None:
                        durations.append(d)
            if durations:
                est_per_run = sum(durations) / len(durations)
                source = "measured"
            else:
                est_per_run = _DEFAULT_STAGE_SECONDS[stage]
                source = "default"
            n_runs = len(seeds) * n_methods
            rows.append(
                {
                    "problem": problem,
                    "stage": stage,
                    "n_runs": n_runs,
                    "est_per_run_s": est_per_run,
                    "est_total_s": est_per_run * n_runs,
                    "source": source,
                }
            )

    total_s = sum(r["est_total_s"] for r in rows)
    summary = {
        "name": recipe.name,
        "total_seconds": total_s,
        "total_hours": total_s / 3600.0,
        "rows": rows,
    }
    # Pretty-print to stdout.
    print(f"[budget] experiment={recipe.name}", flush=True)
    print(
        f"  {'problem':<10s} {'stage':<10s} {'runs':>4s} {'per_run':>9s} {'total':>9s}  src",
        flush=True,
    )
    for r in rows:
        print(
            f"  {r['problem']:<10s} {r['stage']:<10s} {r['n_runs']:>4d} "
            f"{r['est_per_run_s']:>8.1f}s {r['est_total_s']:>8.1f}s  {r['source']}",
            flush=True,
        )
    print(f"  TOTAL: {total_s:.1f}s = {summary['total_hours']:.2f} hours", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Helpers + CLI entrypoints.
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""
    return out


def main_run_cli(argv: list[str] | None = None) -> int:
    """`neuroco experiment <yaml>` entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(prog="neuroco experiment")
    parser.add_argument("yaml", type=Path, help="Path to an experiment YAML recipe.")
    ns = parser.parse_args(argv)
    return run_experiment(ns.yaml)


def main_budget_cli(argv: list[str] | None = None) -> int:
    """`neuroco budget <yaml>` entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(prog="neuroco budget")
    parser.add_argument("yaml", type=Path, help="Path to an experiment YAML recipe.")
    ns = parser.parse_args(argv)
    estimate_budget(ns.yaml)
    return 0
