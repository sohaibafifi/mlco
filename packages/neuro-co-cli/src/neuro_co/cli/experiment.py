"""Validated, isolated experiment recipes with dedicated stage runners.

Every experiment owns ``experiments/<name>/runs`` and a manifest that records
pending, running, complete, and failed stages. Plain training/probing recipes
support Torch environment providers; baseline/eval/explain use the shared
TSP/CVRP protocol. Historical Hydra overrides are rejected explicitly.
"""

from __future__ import annotations

import html
import json
import logging
import re
import subprocess
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

from .artifacts import atomic_write_json, write_run

log = logging.getLogger(__name__)
_STAGE_ORDER = ("train", "baseline", "eval", "explain", "probe")
_DEFAULT_STAGE_SECONDS = {
    "train": 60.0,
    "baseline": 30.0,
    "eval": 30.0,
    "explain": 20.0,
    "probe": 15.0,
}


@dataclass
class ExperimentRecipe:
    name: str
    description: str
    parallel: int
    problems: dict[str, dict[str, Any]]
    raw: dict[str, Any]


@dataclass
class PlannedStage:
    problem: str
    seed: int
    kind: str
    method: str
    options: dict[str, Any]
    training: dict[str, Any]

    @property
    def key(self) -> str:
        suffix = f"_{self.method}" if self.method else ""
        return f"{self.problem}/{self.kind}_seed{self.seed}{suffix}"


def _integer(value: Any, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SystemExit(f"{label} must be an integer >= {minimum}")
    return value


def load_recipe(path: Path) -> ExperimentRecipe:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install neuro-co-cli[report] to use YAML experiments") from exc
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise SystemExit("experiment YAML must be a mapping")
    unknown = set(raw) - {"name", "description", "parallel", "problems"}
    if unknown:
        raise SystemExit(f"Unknown experiment options: {sorted(unknown)}")
    name = str(raw.get("name") or Path(path).stem)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise SystemExit("Experiment name must be a safe directory name")
    parallel = _integer(raw.get("parallel", 1), "parallel")
    problems = raw.get("problems")
    if not isinstance(problems, dict) or not problems:
        raise SystemExit("experiment YAML must declare at least one problem under `problems:`")
    for problem, config in problems.items():
        if not isinstance(problem, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", problem):
            raise SystemExit("Problem names must be lowercase identifiers")
        if not isinstance(config, dict):
            raise SystemExit(f"problem {problem!r} entry must be a mapping")
        unknown = set(config) - {"seeds", *_STAGE_ORDER}
        if unknown:
            raise SystemExit(f"problem {problem!r}: unknown stage {sorted(unknown)}")
        seeds = config.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise SystemExit(f"problem {problem!r} must declare a non-empty `seeds:` list")
        for seed in seeds:
            _integer(seed, "seed", 0)
        if len(set(seeds)) != len(seeds):
            raise SystemExit(f"problem {problem!r}: duplicate seeds")
        for stage in _STAGE_ORDER:
            if stage in config and not isinstance(config[stage], dict):
                raise SystemExit(f"stage {stage!r} block must be a mapping")
    return ExperimentRecipe(name, str(raw.get("description") or ""), parallel, problems, raw)


def _seeds(config: dict[str, Any]) -> list[int]:
    return list(config["seeds"])


def _stage(config: dict[str, Any], stage: str) -> dict[str, Any] | None:
    return config.get(stage)


def _build_plan(recipe: ExperimentRecipe) -> list[PlannedStage]:
    from neuro_co.cli.runners import TrainArgs
    from neuro_co.cli.training import TrainingConfig
    from neuro_co.core.env_registry import available_envs

    if recipe.parallel != 1:
        raise SystemExit("The experiment executor currently requires parallel: 1")
    train_keys = {field.name for field in fields(TrainArgs)} - {
        "problem",
        "seed",
        "out_dir",
        "resume",
    }
    explain_keys = {
        "methods",
        "num_instances",
        "top_k",
        "max_steps",
        "ig_steps",
        "ig_baseline",
        "run_sufficiency",
        "run_sanity_check",
        "sanity_trials",
    }
    probe_keys = {
        "num_instances",
        "epochs",
        "val_frac",
        "lr",
        "pca_components",
        "ica_components",
        "no_figures",
    }
    result = []
    for problem, config in recipe.problems.items():
        if "train" not in config:
            raise SystemExit(
                f"{problem}: recipes must declare train; resume reuses completed stages"
            )
        unknown = set(config["train"]) - train_keys
        if unknown:
            raise SystemExit(
                f"{problem}/train: unknown options {sorted(unknown)}; Hydra overrides are unsupported"
            )
        for seed in _seeds(config):
            try:
                args = TrainArgs(problem=problem, seed=seed, **config["train"])
                if problem not in available_envs(backend=args.backend):
                    raise ValueError(f"No {args.backend} environment provider for {problem}")
                for key in (
                    "epochs",
                    "steps_per_epoch",
                    "batch_size",
                    "eval_batch_size",
                    "size",
                    "hidden_dim",
                    "num_layers",
                    "num_heads",
                ):
                    _integer(getattr(args, key), key)
                if args.backend not in {"torch", "jax"}:
                    raise ValueError("backend must be torch or jax")
                if args.algo not in {"pomo", "reinforce", "ppo"}:
                    raise ValueError("algo must be pomo, reinforce or ppo")
                if args.backbone not in {"am", "gnn", "matnet", "mamba"}:
                    raise ValueError("Unknown backbone")
                if args.precision != "fp32":
                    raise ValueError("The CLI requires fp32 precision")
                if args.hidden_dim % args.num_heads:
                    raise ValueError("num_heads must divide hidden_dim")
                if args.checkpoint_every_steps is not None:
                    _integer(args.checkpoint_every_steps, "checkpoint_every_steps")
                shared = problem in {"tsp", "cvrp"} and (args.algo, args.backbone) == ("pomo", "am")
                if shared:
                    TrainingConfig(
                        **{
                            field.name: getattr(args, field.name)
                            for field in fields(TrainingConfig)
                        }
                    )
                elif args.backend != "torch":
                    raise ValueError("JAX requires TSP/CVRP, POMO and AM")
                if any(stage in config for stage in ("baseline", "eval", "explain")) and not shared:
                    raise ValueError(
                        "baseline/eval/explain recipes require TSP/CVRP with POMO and AM"
                    )
                if args.backend != "torch" and any(
                    stage in config for stage in ("explain", "probe")
                ):
                    raise ValueError("explain/probe require the Torch backend")
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"{problem}/train: {exc}") from exc
            training = asdict(args)
            for kind in _STAGE_ORDER:
                if kind not in config:
                    continue
                options = dict(config[kind])
                allowed = (
                    train_keys
                    if kind == "train"
                    else explain_keys
                    if kind == "explain"
                    else probe_keys
                    if kind == "probe"
                    else set()
                )
                unknown = set(options) - allowed
                if unknown:
                    raise SystemExit(f"{problem}/{kind}: unknown options {sorted(unknown)}")
                if kind == "explain":
                    methods = options.pop("methods", ["gradient"])
                    if (
                        not isinstance(methods, list)
                        or not methods
                        or any(
                            method not in {"gradient", "ig", "contrastive"} for method in methods
                        )
                    ):
                        raise SystemExit("explain.methods must list gradient, ig or contrastive")
                    if len(set(methods)) != len(methods):
                        raise SystemExit("explain.methods must be unique")
                    for key in ("num_instances", "top_k", "ig_steps", "sanity_trials"):
                        if key in options:
                            _integer(options[key], key)
                    if options.get("max_steps") is not None:
                        _integer(options["max_steps"], "max_steps")
                    if options.get("ig_baseline", "zero") not in {"zero", "mean"}:
                        raise SystemExit("ig_baseline must be zero or mean")
                    for key in ("run_sufficiency", "run_sanity_check"):
                        if key in options and not isinstance(options[key], bool):
                            raise SystemExit(f"{key} must be a boolean")
                    for method in methods:
                        result.append(
                            PlannedStage(
                                problem, seed, kind, method, {"method": method, **options}, training
                            )
                        )
                else:
                    if kind == "probe":
                        from neuro_co.problems import get_bank

                        try:
                            get_bank(problem)
                        except KeyError as exc:
                            raise SystemExit(f"No concept bank for {problem}; omit probe") from exc
                        for key in ("num_instances", "epochs"):
                            if key in options:
                                _integer(options[key], key)
                        for key in ("pca_components", "ica_components"):
                            if key in options:
                                _integer(options[key], key, 0)
                        if not 0 < options.get("val_frac", 0.3) < 1:
                            raise SystemExit("probe.val_frac must be between zero and one")
                        if options.get("lr", 1e-2) <= 0:
                            raise SystemExit("probe.lr must be positive")
                        if "no_figures" in options and not isinstance(options["no_figures"], bool):
                            raise SystemExit("probe.no_figures must be a boolean")
                    result.append(PlannedStage(problem, seed, kind, "", options, training))
    return result


def _validate_execution_plan(recipe: ExperimentRecipe) -> None:
    _build_plan(recipe)


def _execute_stage(stage: PlannedStage, experiment_dir: Path, resume: bool) -> None:
    import numpy as np

    from .runners import TrainArgs, train_run
    from .stage_runners import run_baseline, run_evaluation, run_explanation
    from .training import TrainingConfig, array_digest, batch

    output = experiment_dir / "runs" / stage.key
    train_dir = experiment_dir / "runs" / stage.problem / f"train_seed{stage.seed}"
    args = TrainArgs(**{**stage.training, "out_dir": str(train_dir)})
    checkpoint = train_dir / ("best.pt" if args.backend == "torch" else "best.npz")
    if stage.kind == "train":
        if resume and (train_dir / "latest.pt").is_file():
            args.resume = str(train_dir / "latest.pt")
        elif train_dir.exists() and any(train_dir.iterdir()):
            archive = experiment_dir / "attempts" / stage.key
            archive.parent.mkdir(parents=True, exist_ok=True)
            index = 1
            while archive.with_name(f"{archive.name}.{index}").exists():
                index += 1
            train_dir.rename(archive.with_name(f"{archive.name}.{index}"))
        train_run(args)
        return
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint}")
    if stage.kind == "probe":
        from neuro_co.probe.probes_cli import main as probe_main

        argv = [str(train_dir), "--out", str(output), "--seed", str(stage.seed)]
        for key, value in stage.options.items():
            if isinstance(value, bool):
                if value:
                    argv.append("--" + key.replace("_", "-"))
            else:
                argv.extend(["--" + key.replace("_", "-"), str(value)])
        probe_main(argv)
        write_run(
            output,
            kind="probe",
            problem=stage.problem,
            seed=stage.seed,
            backend=args.backend,
            config={**stage.training, **stage.options},
            metrics={},
            artifacts={"probes": "probes.json", "report": "probes.md"},
        )
        return
    config = TrainingConfig(
        **{field.name: getattr(args, field.name) for field in fields(TrainingConfig)}
    )
    dataset = experiment_dir / "datasets" / f"{stage.problem}_seed{stage.seed}.npz"
    data = batch(config, "test")
    if dataset.exists():
        with np.load(dataset, allow_pickle=False) as archive:
            saved = {key: archive[key] for key in archive.files}
        if array_digest(saved) != array_digest(data):
            raise ValueError(f"Saved test dataset changed: {dataset}")
    else:
        dataset.parent.mkdir(parents=True, exist_ok=True)
        temporary = dataset.with_suffix(".tmp.npz")
        np.savez(temporary, **cast(Any, data))
        temporary.replace(dataset)
    if stage.kind == "baseline":
        run_baseline(output, config, args.backend, dataset)
    elif stage.kind == "eval":
        baseline = (
            experiment_dir / "runs" / stage.problem / f"baseline_seed{stage.seed}" / "baseline.json"
        )
        run_evaluation(
            output,
            config,
            args.backend,
            args.device,
            checkpoint,
            dataset,
            baseline if baseline.is_file() else None,
        )
    else:
        run_explanation(output, config, args.backend, args.device, checkpoint, stage.options)


def _validate_completed_stages(
    plan: list[PlannedStage], experiment_dir: Path, manifest: dict[str, Any]
) -> None:
    """Reject incomplete or corrupted receipts before changing any campaign state."""
    import numpy as np

    from .artifacts import RUN_SCHEMA, RUN_STATUSES
    from .training import TrainingConfig, array_digest, batch

    completed = []
    for stage in plan:
        entry = manifest["stages"][stage.key]
        if not isinstance(entry, dict) or entry.get("status") not in RUN_STATUSES:
            raise ValueError(f"Invalid experiment stage status: {stage.key}")
        if entry["status"] != "complete":
            continue
        output = experiment_dir / "runs" / stage.key
        run_path = output / "run.json"
        try:
            record = json.loads(run_path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Completed stage has missing or invalid run metadata: {stage.key}"
            ) from exc
        identity = {
            "schema": RUN_SCHEMA,
            "kind": stage.kind,
            "problem": stage.problem,
            "seed": stage.seed,
            "backend": stage.training["backend"],
            "status": "complete",
        }
        if not isinstance(record, dict) or any(
            record.get(key) != value for key, value in identity.items()
        ):
            raise ValueError(f"Completed stage metadata identity changed: {stage.key}")
        if not isinstance(record.get("run_id"), str) or not record["run_id"]:
            raise ValueError(f"Completed stage has no run identity: {stage.key}")
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError(f"Completed stage has no artifact references: {stage.key}")
        required = {
            "train": ["config.json", "metrics.json", "best.pt", "latest.pt"]
            if stage.training["backend"] == "torch"
            else ["config.json", "metrics.json", "best.npz", "checkpoint.npz"],
            "baseline": ["baseline.json"],
            "eval": ["evaluation.json"],
            "explain": ["explanation.json", "instances.pt"],
            "probe": ["probes.json", "probes.md"],
        }[stage.kind]
        for filename in required:
            if not (output / filename).is_file():
                raise ValueError(
                    f"Completed stage is missing required artifact: {output / filename}"
                )
        for reference in artifacts.values():
            if not isinstance(reference, str):
                raise ValueError(f"Invalid artifact reference in {run_path}")
            target = (output / reference).resolve()
            if not target.is_relative_to(experiment_dir.resolve()) or not target.is_file():
                raise ValueError(f"Completed stage has a missing or external artifact: {target}")
        completed.append(stage)

    checked: set[tuple[str, int]] = set()
    for stage in completed:
        if stage.kind not in {"baseline", "eval", "explain"}:
            continue
        training_key = f"{stage.problem}/train_seed{stage.seed}"
        if manifest["stages"][training_key]["status"] != "complete":
            raise ValueError(f"Completed stage has an incomplete training dependency: {stage.key}")
        config = TrainingConfig(
            **{field.name: stage.training[field.name] for field in fields(TrainingConfig)}
        )
        expected_hash = array_digest(batch(config, "test"))
        dataset_key = (stage.problem, stage.seed)
        if dataset_key not in checked:
            dataset = experiment_dir / "datasets" / f"{stage.problem}_seed{stage.seed}.npz"
            try:
                with np.load(dataset, allow_pickle=False) as archive:
                    data = {key: archive[key] for key in archive.files}
                actual_hash = array_digest(data)
            except (OSError, ValueError, EOFError) as exc:
                raise ValueError(f"Saved test dataset is missing or corrupted: {dataset}") from exc
            if actual_hash != expected_hash:
                raise ValueError(f"Saved test dataset hash changed: {dataset}")
            checked.add(dataset_key)
        receipts = [experiment_dir / "runs" / training_key / "evaluation.json"]
        if stage.kind in {"baseline", "eval"}:
            filename = "baseline.json" if stage.kind == "baseline" else "evaluation.json"
            receipts.append(experiment_dir / "runs" / stage.key / filename)
        for receipt in receipts:
            try:
                data = json.loads(receipt.read_text())
            except (OSError, ValueError) as exc:
                raise ValueError(f"Missing or invalid test receipt: {receipt}") from exc
            if (
                not isinstance(data, dict)
                or data.get("test_sha256") != expected_hash
                or data.get("num_instances") != config.eval_batch_size
                or not isinstance(data.get("costs"), list)
                or len(data["costs"]) != config.eval_batch_size
            ):
                raise ValueError(f"Test receipt no longer matches the fixed corpus: {receipt}")


def _write_report(experiment_dir: Path, manifest: dict[str, Any], table: Any) -> None:
    rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{html.escape(value['status'])}</td></tr>"
        for key, value in manifest["stages"].items()
    )
    links = "".join(
        f"<li><a href='{html.escape(path, quote=True)}'>{html.escape(path)}</a></li>"
        for path in [
            "manifest.json",
            "results.csv",
            "results.parquet",
            *manifest.get("figures", []),
        ]
    )
    content = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        f"<title>{html.escape(manifest['name'])}</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}table{border-collapse:collapse;margin:24px 0}td,th{padding:8px 12px;border:1px solid #ddd;text-align:left}</style>"
        f"<h1>{html.escape(manifest['name'])}</h1><p>{html.escape(manifest['description'])}</p>"
        f"<p>Status: {html.escape(manifest['status'])}. Baselines are heuristics, not optimality certificates.</p>"
        "<h2>Stages</h2><table><tr><th>Stage</th><th>Status</th></tr>" + rows + "</table>"
        f"<h2>Artifacts</h2><ul>{links}</ul><h2>Metrics</h2>"
        + table.to_html(index=False, escape=True)
        + "</html>"
    )
    (experiment_dir / "report.html").write_text(content)


def run_experiment(
    yaml_path: Path,
    *,
    out_dir: Path | None = None,
    resume: bool = False,
    retry_failed: bool = False,
    dry_run: bool = False,
) -> int:
    from .figures import render_all
    from .results import collect, write_table

    recipe = load_recipe(yaml_path)
    plan = _build_plan(recipe)
    experiment_dir = Path(out_dir) if out_dir is not None else Path("experiments") / recipe.name
    if dry_run:
        print(
            json.dumps(
                {
                    "name": recipe.name,
                    "output": str(experiment_dir),
                    "stages": [stage.key for stage in plan],
                },
                indent=2,
            )
        )
        return 0
    # Fail before training if optional report dependencies are unavailable.
    try:
        import matplotlib  # noqa: F401
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Install neuro-co-cli[report] to execute experiments and write reports"
        ) from exc

    manifest_path = experiment_dir / "manifest.json"
    continuing = resume or retry_failed
    if continuing:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No experiment manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("schema") != "neuro-co-experiment/v1"
            or manifest.get("recipe") != recipe.raw
        ):
            raise ValueError("Cannot resume with a changed recipe or unsupported manifest")
        if set(manifest.get("stages", {})) != {stage.key for stage in plan}:
            raise ValueError("Experiment manifest stages do not match the recipe")
        _validate_completed_stages(plan, experiment_dir, manifest)
        if not retry_failed and any(
            item["status"] == "failed" for item in manifest["stages"].values()
        ):
            raise ValueError("Experiment contains failed stages; use --retry-failed")
    else:
        if experiment_dir.exists() and any(experiment_dir.iterdir()):
            raise FileExistsError(f"{experiment_dir} is not empty; use --resume or a new --out-dir")
        manifest = {
            "schema": "neuro-co-experiment/v1",
            "name": recipe.name,
            "description": recipe.description,
            "yaml_path": str(yaml_path),
            "started_at": _now_iso(),
            "git_sha": _git_sha(),
            "recipe": recipe.raw,
            "stages": {
                stage.key: {"status": "pending", "attempts": 0, "path": f"runs/{stage.key}"}
                for stage in plan
            },
        }
    manifest["status"] = "running"
    atomic_write_json(manifest_path, manifest)
    for stage in plan:
        entry = manifest["stages"][stage.key]
        output = experiment_dir / "runs" / stage.key
        if entry["status"] == "complete":
            continue
        entry.update(status="running", started_at=_now_iso(), attempts=entry["attempts"] + 1)
        entry.pop("error", None)
        atomic_write_json(manifest_path, manifest)
        try:
            _execute_stage(stage, experiment_dir, continuing)
            entry.update(status="complete", finished_at=_now_iso())
        except KeyboardInterrupt:
            # Retain a running stage so --resume can continue from its checkpoint.
            entry["interrupted_at"] = _now_iso()
            atomic_write_json(manifest_path, manifest)
            raise
        except BaseException as exc:
            entry.update(
                status="failed", finished_at=_now_iso(), error=f"{type(exc).__name__}: {exc}"
            )
            manifest["status"] = "failed"
            atomic_write_json(manifest_path, manifest)
            previous_path = output / "run.json"
            previous = json.loads(previous_path.read_text()) if previous_path.is_file() else {}
            write_run(
                output,
                kind=stage.kind,
                problem=stage.problem,
                seed=stage.seed,
                backend=stage.training["backend"],
                config=previous.get("config", {**stage.training, **stage.options}),
                metrics=previous.get("metrics", {}),
                artifacts=previous.get("artifacts", {}),
                status="failed",
                error=entry["error"],
            )
            raise
        atomic_write_json(manifest_path, manifest)
    table = collect(experiment_dir / "runs")
    write_table(table, experiment_dir / "results.parquet")
    write_table(table, experiment_dir / "results.csv")
    figures = render_all(table, experiment_dir / "figs")
    manifest.update(
        status="complete",
        finished_at=_now_iso(),
        n_rows=len(table),
        results="results.parquet",
        report="report.html",
        figures=[str(path.relative_to(experiment_dir)) for path in figures],
    )
    _write_report(experiment_dir, manifest, table)
    atomic_write_json(manifest_path, manifest)
    print(
        f"[experiment] {recipe.name} -> {experiment_dir} (rows={len(table)}, figures={len(figures)})",
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
    """``neuroco experiment <yaml>`` entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(prog="neuroco experiment")
    parser.add_argument("yaml", type=Path, help="Path to an experiment YAML recipe.")
    parser.add_argument(
        "--out-dir", type=Path, help="Experiment directory; default experiments/<name>."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Continue interrupted work and skip completed stages."
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed stages and continue pending stages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the execution plan without writing files.",
    )
    ns = parser.parse_args(argv)
    options = {}
    if ns.out_dir is not None:
        options["out_dir"] = ns.out_dir
    for name in ("resume", "retry_failed", "dry_run"):
        if getattr(ns, name):
            options[name] = True
    return run_experiment(ns.yaml, **options)


def main_budget_cli(argv: list[str] | None = None) -> int:
    """`neuroco budget <yaml>` entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(prog="neuroco budget")
    parser.add_argument("yaml", type=Path, help="Path to an experiment YAML recipe.")
    ns = parser.parse_args(argv)
    estimate_budget(ns.yaml)
    return 0
