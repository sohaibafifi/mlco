"""Compare stored policy and classical-baseline costs.

Read `energy_eval.json` and `energy_baseline.json` from sibling run directories.
For minimization, the relative gap is `(abs(policy) - abs(baseline)) /
abs(baseline)`; orienteering reverses the numerator. The configurable threshold
marks a gap as acceptable. It is not a proof of optimization convergence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ConvergenceRow:
    problem: str
    seed: int
    run_dir: str
    policy_cost: float | None
    baseline_cost: float | None
    gap: float | None
    converged: bool  # gap <= threshold


# Problems we maximise (reward is prize, higher = better). Everything
# else (routing, scheduling) is a minimisation -- lower cost = better.
_MAXIMISATION_PROBLEMS: frozenset[str] = frozenset({"op"})


def read_convergence(
    run_dirs: list[Path],
    *,
    threshold: float = 0.15,
) -> list[ConvergenceRow]:
    """Tabulate (policy_cost, baseline_cost, gap) per run dir.

    Gap convention: positive means policy is *worse* than the
    classical baseline, negative means policy beats it. For
    minimisation problems gap = `(policy - baseline) / baseline`.
    For maximisation problems (OP) we flip the numerator so the
    sign rule stays consistent (positive = worse).
    """
    rows: list[ConvergenceRow] = []
    for rd in run_dirs:
        problem = _infer_problem_from_path(rd)
        seed = _infer_seed_from_path(rd)
        eval_dir = _sibling(rd, "eval")
        base_dir = _sibling(rd, "baseline")
        p_cost = _read_cost(eval_dir / "energy_eval.json") if eval_dir else None
        b_cost = _read_cost(base_dir / "energy_baseline.json") if base_dir else None
        gap = None
        if p_cost is not None and b_cost is not None and b_cost != 0:
            pa, ba = abs(p_cost), abs(b_cost)
            if problem.lower() in _MAXIMISATION_PROBLEMS:
                gap = float((ba - pa) / ba)  # higher prize = better
            else:
                gap = float((pa - ba) / ba)  # lower cost = better
        converged = gap is not None and gap <= threshold
        rows.append(
            ConvergenceRow(
                problem=problem,
                seed=seed,
                run_dir=str(rd),
                policy_cost=p_cost,
                baseline_cost=b_cost,
                gap=gap,
                converged=converged,
            )
        )
    return rows


def convergence_table_latex(
    rows: list[ConvergenceRow],
    *,
    threshold: float = 0.15,
) -> str:
    """Render mean costs and gaps by problem as a booktabs LaTeX table."""
    # Aggregate by problem: mean gap + worst-case across seeds.
    by_problem: dict[str, list[ConvergenceRow]] = {}
    for r in rows:
        by_problem.setdefault(r.problem, []).append(r)

    lines: list[str] = []
    lines.append(r"\begin{tabular}{lrrrc}")
    lines.append(r"\toprule")
    lines.append(r"problem & policy cost & baseline cost & mean gap & converged \\")
    lines.append(r"\midrule")
    for problem in sorted(by_problem):
        rs = by_problem[problem]
        gaps = [r.gap for r in rs if r.gap is not None]
        if not gaps:
            lines.append(f"{problem} & -- & -- & -- & -- \\\\")
            continue
        mean_gap = sum(gaps) / len(gaps)
        mean_policy = sum(r.policy_cost for r in rs if r.policy_cost is not None) / len(rs)
        mean_baseline = sum(r.baseline_cost for r in rs if r.baseline_cost is not None) / len(rs)
        marker = r"\checkmark" if mean_gap <= threshold else r"\times"
        lines.append(
            f"{problem} & {mean_policy:.3f} & {mean_baseline:.3f} & {mean_gap:+.2%} & {marker} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _read_cost(path: Path) -> float | None:
    """Read the first numeric cost field from a stored energy record.

    Accepted keys are `avg_cost`, `nn_cost`, `policy_cost`, and `cost`, in that
    order. Return None for missing or malformed records.
    """
    if not path.is_file():
        return None
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for key in ("avg_cost", "nn_cost", "policy_cost", "cost"):
        v = d.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _sibling(train_run_dir: Path, stage: str) -> Path | None:
    """Locate a sibling evaluation or baseline directory for a training seed.

    For baselines, fall back to `baseline_seed0`, then the first baseline directory.
    The caller must ensure that a shared baseline uses the matching instance set.
    """
    name = train_run_dir.name
    if not name.startswith("train_seed"):
        return None
    new_name = stage + "_seed" + name[len("train_seed") :]
    sib = train_run_dir.parent / new_name
    if sib.is_dir():
        return sib
    if stage == "baseline":
        # Shared-test-set convention: one baseline run covers all seeds.
        fallback = train_run_dir.parent / "baseline_seed0"
        if fallback.is_dir():
            return fallback
        cands = sorted(train_run_dir.parent.glob("baseline_seed*"))
        if cands:
            return cands[0]
    return None


def _infer_problem_from_path(rd: Path) -> str:
    # outputs/<problem>/train_seed<N>
    if len(rd.parts) >= 2:
        return rd.parts[-2]
    return "unknown"


def _infer_seed_from_path(rd: Path) -> int:
    name = rd.name
    if name.startswith("train_seed"):
        digits = name[len("train_seed") :].split("_")[0]
        try:
            return int(digits)
        except ValueError:
            return -1
    return -1


def main(argv: list[str] | None = None) -> int:
    """`neuroco-cax-convergence` CLI -- table + JSON dump."""
    import argparse

    parser = argparse.ArgumentParser(prog="neuroco-cax-convergence")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Optimality-gap threshold below which a policy is converged.",
    )
    parser.add_argument(
        "--out-latex",
        type=Path,
        default=None,
        help="Optional path to write the booktabs LaTeX table.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to dump the per-run rows as JSON.",
    )
    args = parser.parse_args(argv)

    rows = read_convergence(args.run_dirs, threshold=args.threshold)
    print(f"{'problem':>8s} {'seed':>4s} {'policy':>10s} {'baseline':>10s} {'gap':>8s} converged")
    for r in rows:
        p = "--" if r.policy_cost is None else f"{r.policy_cost:>10.3f}"
        b = "--" if r.baseline_cost is None else f"{r.baseline_cost:>10.3f}"
        g = "--" if r.gap is None else f"{r.gap:>+7.2%}"
        c = "yes" if r.converged else "no"
        print(f"{r.problem:>8s} {r.seed:>4d} {p} {b} {g} {c}")

    if args.out_latex:
        args.out_latex.parent.mkdir(parents=True, exist_ok=True)
        args.out_latex.write_text(convergence_table_latex(rows, threshold=args.threshold))
        print(f"[convergence] LaTeX table -> {args.out_latex}")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps([r.__dict__ for r in rows], indent=2))
        print(f"[convergence] JSON -> {args.out_json}")
    return 0


_ = Any  # silence import-style lint when only used in type hints
