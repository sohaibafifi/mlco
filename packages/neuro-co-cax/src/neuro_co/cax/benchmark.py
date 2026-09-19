"""Run constraint-family attribution on saved core policies.

Read `metrics.json` and a checkpoint from each run directory, sample instances,
and write `lambda_attribution.json` or `lambda_attribution_weighted.json`.
Reports include feature columns, mean and per-step family scores, and the
highest-ranked family per decision. `benchmark_runs` can also write aggregate
rows to Parquet when its optional dependencies are installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from neuro_co.cax.lambda_attribution import LambdaAttribution, lambda_attribution


@dataclass
class BenchmarkRow:
    """One row of the aggregate benchmark table."""

    problem: str
    seed: int
    run_dir: str
    constraint: str
    mean_score: float
    num_instances: int
    num_steps: int


def benchmark_run(
    run_dir: Path,
    *,
    num_instances: int = 8,
    max_steps: int | None = 16,
    problem: str | None = None,
    multipliers: dict[str, float] | None = None,
) -> tuple[LambdaAttribution, list[BenchmarkRow]]:
    """Run Lambda-attribution on a single run dir; return (trace, rows).

    Side-effect: writes `lambda_attribution.json` under `run_dir`.

    `multipliers` is an optional `{family: weight}` dict forwarded to
    `lambda_attribution` (None = proxy / equal-weight).
    """
    import torch

    from neuro_co.core.factory import make_env, make_model

    run_dir = Path(run_dir)
    meta = _load_run_meta(run_dir)
    problem = problem or str(meta["problem"])
    seed = int(meta.get("seed", 0))

    env = make_env(problem, size=int(meta.get("size", 50)), **_env_kwargs(problem))
    model = make_model(env, **meta.get("arch", {}))
    ckpt = torch.load(_resolve_ckpt(run_dir), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    device = _pick_device()
    model = model.to(device).eval()
    state = env.reset(num_instances, generator=torch.Generator().manual_seed(seed)).to(device)

    trace = lambda_attribution(
        model,
        env,
        state,
        problem=problem,
        max_steps=max_steps,
        multipliers=multipliers,
    )

    rows = [
        BenchmarkRow(
            problem=problem,
            seed=seed,
            run_dir=str(run_dir),
            constraint=name,
            mean_score=float(trace.scores[..., k_idx].mean().item()),
            num_instances=trace.batch_size,
            num_steps=trace.num_steps,
        )
        for k_idx, name in enumerate(trace.constraint_names)
    ]

    suffix = "_weighted" if multipliers else ""
    out = run_dir / f"lambda_attribution{suffix}.json"
    out.write_text(
        json.dumps(
            {
                "problem": problem,
                "seed": seed,
                "multipliers_mode": "weighted" if multipliers else "proxy",
                "multipliers_values": (
                    trace.multipliers.tolist() if trace.multipliers is not None else None
                ),
                "num_instances": trace.batch_size,
                "max_steps": trace.num_steps,
                "constraint_names": trace.constraint_names,
                "feature_cols_per_family": [list(cols) for cols in trace.feature_cols_per_family],
                "mean_scores": [
                    float(trace.scores[..., k_idx].mean().item())
                    for k_idx in range(trace.num_families)
                ],
                "per_step_scores": trace.scores.mean(dim=0).tolist(),
                "top_family_per_step": trace.top_family_per_step().tolist(),
            },
            indent=2,
        )
    )
    return trace, rows


def benchmark_runs(
    run_dirs: list[Path],
    *,
    num_instances: int = 8,
    max_steps: int | None = 16,
    out_parquet: Path | None = None,
    multipliers: dict[str, float] | None = None,
) -> Any:
    """Loop over run dirs, run benchmark, optionally write aggregate parquet."""
    all_rows: list[BenchmarkRow] = []
    failures: list[tuple[Path, str]] = []
    for rd in run_dirs:
        try:
            _trace, rows = benchmark_run(
                rd,
                num_instances=num_instances,
                max_steps=max_steps,
                multipliers=multipliers,
            )
            all_rows.extend(rows)
        except Exception as exc:  # surface per-run, keep going
            failures.append((rd, f"{type(exc).__name__}: {exc}"))

    if failures:
        for rd, msg in failures:
            print(f"[cax-benchmark] FAIL {rd}: {msg}")

    if out_parquet is not None and all_rows:
        try:
            import pandas as pd

            df = pd.DataFrame([row.__dict__ for row in all_rows])
            out_parquet.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_parquet, index=False)
            print(f"[cax-benchmark] wrote {out_parquet} ({len(df)} rows)")
            return df
        except ImportError:
            print("[cax-benchmark] pandas not available, skipping parquet")

    return all_rows


# ---------------------------------------------------------------------------
# Core run-dir helpers read the convention written by the `neuroco` CLI:
# `<run_dir>/metrics.json` (problem + arch in `args`) + `<run_dir>/best.pt`.
# ---------------------------------------------------------------------------


def load_run(run_dir: Path, *, device: Any = None) -> tuple[Any, Any, dict[str, Any]]:
    """Rebuild (env, model, meta) from a `neuroco train` run dir.

    Reads `metrics.json` for problem + arch, builds env + model via
    `core.factory`, and restores `best.pt` (falls back to `latest.pt`).
    The model is moved to `device` (auto-picked if None) and set to eval.
    """
    from neuro_co.core.factory import make_env, make_model

    run_dir = Path(run_dir)
    meta = _load_run_meta(run_dir)
    problem = str(meta["problem"])
    env = make_env(problem, size=int(meta.get("size", 50)), **_env_kwargs(problem))
    model = make_model(env, **meta.get("arch", {}))
    ckpt = torch.load(_resolve_ckpt(run_dir), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    dev = device or _pick_device()
    model = model.to(dev).eval()
    return env, model, meta


def _load_run_meta(run_dir: Path) -> dict[str, Any]:
    """Read `metrics.json` and return the flattened training args.

    Expects the run dir produced by `neuroco train`: a `metrics.json` whose
    `args` block carries `problem`, `size`, and the model arch.
    """
    p = Path(run_dir) / "metrics.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"No metrics.json under {run_dir}; pass a directory produced by `neuroco train`."
        )
    args = json.loads(p.read_text()).get("args", {})
    arch = {k: args[k] for k in ("backbone", "hidden_dim", "num_layers", "num_heads") if k in args}
    return {
        "problem": args.get("problem"),
        "size": args.get("size", 50),
        "seed": args.get("seed", 0),
        "arch": arch,
    }


def _env_kwargs(problem: str) -> dict[str, Any]:
    """Per-problem env-constructor defaults beyond `size` (match the CLI)."""
    extra: dict[str, dict[str, Any]] = {
        "cvrp": {"capacity": 50.0},
        "cvrptw": {"capacity": 50.0, "horizon": 10.0, "window_width": 2.0},
        "vrptw": {"capacity": 50.0, "horizon": 10.0, "window_width": 2.0},
        "op": {"budget": 3.0},
        "mtsp": {"num_agents": 5},
        "fjsp": {"ops_per_job": 3, "num_machines": 5},
    }
    return extra.get(problem.lower(), {})


def _resolve_ckpt(run_dir: Path, override: str | None = None) -> str:
    if override:
        return str(override)
    for name in ("best.pt", "latest.pt"):
        cand = Path(run_dir) / name
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(f"No best.pt / latest.pt under {run_dir}")


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
