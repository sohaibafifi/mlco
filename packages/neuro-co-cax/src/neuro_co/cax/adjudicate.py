"""Compare constraint attribution with sampled counterfactual families.

For each decoding cell, compare an attribution's highest-ranked family with
the family selected by the feasible counterfactual search. Agreement measures
consistency with that search and depends on its budget and feasibility mode.

The runner loads `metrics.json` and `best.pt` from a core training directory.
Multiplier modes include equal weights (`proxy`), LP weights (`lp`), and
heuristic updates (`subgrad`). Solver-backed paths use optional dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from neuro_co.cax.benchmark import _env_kwargs, _load_run_meta, _pick_device, _resolve_ckpt
from neuro_co.cax.constraint_map import get_constraints
from neuro_co.cax.cp_counterfactual import cp_counterfactual
from neuro_co.cax.lambda_attribution import lambda_attribution

# Which constraint family each perturbable state field belongs to, per problem.
# Mirrors the column groups in `constraint_map` mapped onto the instance
# state fields `cp_counterfactual` perturbs.
FAMILY_OF_FIELD: dict[str, dict[str, str]] = {
    "cvrptw": {"demand": "capacity", "tw_early": "time_window", "tw_late": "time_window"},
    "vrptw": {"demand": "capacity", "tw_early": "time_window", "tw_late": "time_window"},
    "op": {"prize": "prize"},
    "fjsp": {"proc_times": "processing"},
}

# Per-problem scalar epsilon for the counterfactual L-infinity ball.
DEFAULT_EPS_PER_PROBLEM: dict[str, float] = {
    "cvrptw": 0.1,
    "vrptw": 0.1,
    "op": 0.3,
    "fjsp": 5.0,
}


def _resolve_multipliers(mode: str, env, state, problem: str) -> dict[str, float] | None:
    """proxy -> None (equal weight); lp/subgrad -> per-family duals dict."""
    if mode == "proxy":
        return None
    from neuro_co.cax.duals import get_multipliers, state_to_instance

    instance = state_to_instance(state, env, problem, batch_idx=0)
    try:
        return get_multipliers(problem, instance, method=mode)
    except (NotImplementedError, ValueError, RuntimeError):
        return None


def _top_family_per_step(
    mode: str, model, env, state, *, problem: str, max_steps: int
) -> torch.Tensor:
    """`[B, T]` top-1 constraint family per cell for a lambda mode."""
    mult = _resolve_multipliers(mode, env, state, problem)
    attr = lambda_attribution(
        model, env, state, problem=problem, max_steps=max_steps, multipliers=mult
    )
    return attr.top_family_per_step()


@dataclass
class AdjudicationRow:
    problem: str
    seed: int
    run_dir: str
    mode: str
    mean_agreement: float
    n_flipped_steps: int
    n_total_steps: int


@dataclass
class AdjudicationReport:
    problem: str
    seed: int
    constraint_names: list[str]
    modes: list[str]
    lambda_top_per_step: dict[str, torch.Tensor]
    cf_flipped_per_step: torch.Tensor
    cf_top_family_per_step: torch.Tensor
    agreement_per_mode_per_step: dict[str, torch.Tensor]
    mean_agreement_per_mode: dict[str, float] = field(default_factory=dict)


def adjudicate_run(
    run_dir: Path,
    *,
    modes: tuple[str, ...] = ("proxy", "lp", "subgrad"),
    num_instances: int = 4,
    max_steps: int = 4,
    cf_shots: int = 32,
    epsilon: float | None = None,
    seed: int = 0,
    problem: str | None = None,
    feasibility_mode: str = "arithmetic",
) -> tuple[AdjudicationReport, list[AdjudicationRow]]:
    """Run lambda modes + CP-counterfactual on one run dir; write `adjudication.json`."""
    from neuro_co.core.factory import make_env, make_model

    run_dir = Path(run_dir)
    meta = _load_run_meta(run_dir)
    problem = problem or str(meta["problem"])
    seed_meta = int(meta.get("seed", 0))
    eps = epsilon if epsilon is not None else DEFAULT_EPS_PER_PROBLEM.get(problem.lower(), 0.1)
    constraint_names = [name for name, _ in get_constraints(problem)]

    env = make_env(problem, size=int(meta.get("size", 50)), **_env_kwargs(problem))
    model = make_model(env, **meta.get("arch", {}))
    ckpt = torch.load(_resolve_ckpt(run_dir), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    device = _pick_device()
    model = model.to(device).eval()
    state = env.reset(num_instances, generator=torch.Generator().manual_seed(seed)).to(device)

    lambda_top = {
        m: _top_family_per_step(m, model, env, state, problem=problem, max_steps=max_steps)
        for m in modes
    }

    cf = cp_counterfactual(
        model,
        env,
        state,
        problem=problem,
        epsilon=eps,
        max_shots=cf_shots,
        max_steps=max_steps,
        seed=seed,
        feasibility_mode=feasibility_mode,
    )

    # CF top family per flipped cell = family carrying the largest delta mass.
    # cf.delta: {state_field: [T, B, *field]}; map field -> family.
    cf_flipped = cf.flipped.cpu()
    cf_top = torch.full(cf_flipped.shape, -1, dtype=torch.long)
    field_fam = FAMILY_OF_FIELD.get(problem.lower(), {})
    name_idx = {n: i for i, n in enumerate(constraint_names)}
    for b in range(cf_flipped.shape[0]):
        for t in range(cf_flipped.shape[1]):
            if not bool(cf_flipped[b, t]):
                continue
            mass: dict[str, float] = {}
            for fld, fam in field_fam.items():
                if fld in cf.delta:
                    mass[fam] = mass.get(fam, 0.0) + float(cf.delta[fld][t, b].abs().sum())
            if mass:
                best = max(mass, key=lambda k: mass[k])
                cf_top[b, t] = name_idx.get(best, -1)

    agree_per_mode: dict[str, torch.Tensor] = {}
    mean_agree: dict[str, float] = {}
    for m in modes:
        top = lambda_top[m].cpu()
        t = min(top.shape[1], cf_top.shape[1])
        match = (top[:, :t] == cf_top[:, :t]) & cf_flipped[:, :t]
        agree_per_mode[m] = match
        n_flips = int(cf_flipped[:, :t].sum())
        mean_agree[m] = float(int(match.sum()) / n_flips) if n_flips else 0.0

    report = AdjudicationReport(
        problem=problem,
        seed=seed_meta,
        constraint_names=constraint_names,
        modes=list(modes),
        lambda_top_per_step=lambda_top,
        cf_flipped_per_step=cf_flipped,
        cf_top_family_per_step=cf_top,
        agreement_per_mode_per_step=agree_per_mode,
        mean_agreement_per_mode=mean_agree,
    )

    (run_dir / "adjudication.json").write_text(
        json.dumps(
            {
                "problem": problem,
                "seed": seed_meta,
                "constraint_names": constraint_names,
                "modes": list(modes),
                "lambda_top_family_per_step": {m: lambda_top[m].cpu().tolist() for m in modes},
                "cf_flipped_per_step": cf_flipped.int().tolist(),
                "cf_top_family_per_step": cf_top.tolist(),
                "mean_agreement_per_mode": mean_agree,
                "n_flipped_steps": int(cf_flipped.sum()),
                "n_total_steps": int(cf_flipped.numel()),
            },
            indent=2,
        )
    )

    rows = [
        AdjudicationRow(
            problem=problem,
            seed=seed_meta,
            run_dir=str(run_dir),
            mode=m,
            mean_agreement=mean_agree[m],
            n_flipped_steps=int(cf_flipped.sum()),
            n_total_steps=int(cf_flipped.numel()),
        )
        for m in modes
    ]
    return report, rows


def adjudicate_runs(
    run_dirs: list[Path],
    *,
    modes: tuple[str, ...] = ("proxy", "lp", "subgrad"),
    num_instances: int = 4,
    max_steps: int = 4,
    cf_shots: int = 32,
    epsilon: float | None = None,
    seed: int = 0,
    out_parquet: Path | None = None,
    feasibility_mode: str = "arithmetic",
) -> Any:
    """Loop over run dirs; optionally write an aggregate parquet."""
    all_rows: list[AdjudicationRow] = []
    for rd in run_dirs:
        try:
            _, rows = adjudicate_run(
                rd,
                modes=modes,
                num_instances=num_instances,
                max_steps=max_steps,
                cf_shots=cf_shots,
                epsilon=epsilon,
                seed=seed,
                feasibility_mode=feasibility_mode,
            )
            all_rows.extend(rows)
        except Exception as exc:  # surface per-run, keep going
            print(f"[cax-adjudicate] FAIL {rd}: {type(exc).__name__}: {exc}")

    if out_parquet is not None and all_rows:
        try:
            import pandas as pd

            df = pd.DataFrame([r.__dict__ for r in all_rows])
            out_parquet.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_parquet, index=False)
            return df
        except ImportError:
            print("[cax-adjudicate] pandas not available, skipping parquet")
    return all_rows
