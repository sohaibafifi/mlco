"""Paired tests and bootstrap intervals for experiment results.

Three primitives + one CLI subcommand:

- `paired_wilcoxon(a, b)` -> `(W, p_two_sided)`. Wilcoxon
  signed-rank for two paired samples (e.g. method A vs B across
  seeds on the same problem). No SciPy dependency: a vectorised
  re-implementation with exact-distribution fallback for small N.
- `bootstrap_ci(x, statistic=mean, n=10_000, alpha=0.05)` ->
  `(lo, hi)`. Percentile bootstrap CI around a callable statistic.
- `compare_methods(df, metric)` -> long-format DataFrame with one
  row per (problem, method-pair) holding mean delta, Wilcoxon p,
  bootstrap CI bounds.

CLI: `neuroco stats <results.parquet> --metric <m>` prints the
table; `--out <csv>` writes it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Wilcoxon signed-rank test on paired samples `a` and `b`.

    Returns `(W, p_two_sided)`. Pairs with `a[i] == b[i]` are
    dropped (standard convention). For `n <= 25` the exact null
    distribution is enumerated; above that, the normal approximation
    with continuity correction is used.
    """
    if len(a) != len(b):
        raise ValueError(f"a and b must have equal length, got {len(a)} vs {len(b)}")
    diffs = [float(x) - float(y) for x, y in zip(a, b, strict=True) if x != y]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0

    # Rank absolute differences (average rank for ties).
    abs_diffs = sorted(((abs(d), i) for i, d in enumerate(diffs)), key=lambda t: t[0])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_diffs[j + 1][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based ranks
        for k in range(i, j + 1):
            ranks[abs_diffs[k][1]] = avg_rank
        i = j + 1

    w_plus = sum(r for r, d in zip(ranks, diffs, strict=True) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, diffs, strict=True) if d < 0)
    w = min(w_plus, w_minus)

    if n <= 25:
        # Exact two-sided p: enumerate all 2^n sign assignments.
        total = 1 << n
        extreme = 0
        for bits in range(total):
            s_plus = sum(ranks[k] for k in range(n) if (bits >> k) & 1)
            if s_plus <= w or s_plus >= (sum(ranks) - w):
                extreme += 1
        return float(w), min(1.0, extreme / total)

    # Normal approximation with continuity correction.
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return float(w), 1.0
    z = (w - mu + 0.5) / sigma
    p = 2 * 0.5 * math.erfc(abs(z) / math.sqrt(2))  # two-sided
    return float(w), float(min(1.0, p))


def bootstrap_ci(
    x: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = lambda v: sum(v) / max(1, len(v)),
    n: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI around `statistic`. No SciPy dep."""
    import random

    rng = random.Random(seed)
    samples = list(x)
    k = len(samples)
    if k == 0:
        return float("nan"), float("nan")
    boots: list[float] = []
    for _ in range(n):
        resample = [samples[rng.randrange(k)] for _ in range(k)]
        boots.append(float(statistic(resample)))
    boots.sort()
    lo = boots[int((alpha / 2) * n)]
    hi = boots[min(n - 1, int((1 - alpha / 2) * n))]
    return lo, hi


def compare_methods(
    df: Any,
    metric: str,
    *,
    kind: str = "explain",
    higher_is_better: bool = True,
    n_bootstrap: int = 2000,
) -> Any:
    """Pairwise method comparison per problem on `metric`.

    Returns a long-format DataFrame with columns:
    `(problem, method_a, method_b, n, delta_mean, ci_lo, ci_hi,
    wilcoxon_W, wilcoxon_p, winner)`.

    `delta = method_a - method_b`, averaged over the seeds that
    appear in both groups for the same problem. `winner` is the
    method whose mean is better given `higher_is_better`, or
    `tie` when the CI brackets zero.
    """
    import pandas as pd

    sub = df[(df["kind"] == kind) & (df["metric"] == metric)]
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "problem",
                "method_a",
                "method_b",
                "n",
                "delta_mean",
                "ci_lo",
                "ci_hi",
                "wilcoxon_W",
                "wilcoxon_p",
                "winner",
            ]
        )

    rows: list[dict[str, Any]] = []
    for problem, prob_df in sub.groupby("problem"):
        methods = sorted(prob_df["method"].unique())
        for a, b in combinations(methods, 2):
            pa = prob_df[prob_df["method"] == a].set_index("seed")["value"]
            pb = prob_df[prob_df["method"] == b].set_index("seed")["value"]
            common = pa.index.intersection(pb.index)
            if len(common) < 2:
                continue
            va = pa.loc[common].values
            vb = pb.loc[common].values
            deltas = [float(va[i]) - float(vb[i]) for i in range(len(common))]
            delta_mean = sum(deltas) / len(deltas)
            ci_lo, ci_hi = bootstrap_ci(deltas, n=n_bootstrap)
            w, p = paired_wilcoxon(va.tolist(), vb.tolist())
            if ci_lo > 0 and ci_hi > 0:
                winner = a if higher_is_better else b
            elif ci_lo < 0 and ci_hi < 0:
                winner = b if higher_is_better else a
            else:
                winner = "tie"
            rows.append(
                {
                    "problem": problem,
                    "method_a": a,
                    "method_b": b,
                    "n": len(common),
                    "delta_mean": delta_mean,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "wilcoxon_W": w,
                    "wilcoxon_p": p,
                    "winner": winner,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main_cli(argv: list[str] | None = None) -> int:
    """`neuroco stats <results.parquet> --metric <m>` entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(prog="neuroco stats")
    parser.add_argument("results", type=Path, help="Path to a results.parquet or .csv.")
    parser.add_argument(
        "--metric",
        default="deletion.mean_flip_rate",
        help="Metric to compare across methods. Default deletion.mean_flip_rate.",
    )
    parser.add_argument(
        "--kind",
        default="explain",
        help="Result kind (default explain).",
    )
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help="Flip the winner-selection direction (default: higher = better).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output csv with the comparison table.",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=2000,
        help="Bootstrap resamples for the CI. Default 2000.",
    )
    ns = parser.parse_args(argv)

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pandas required for `neuroco stats`") from exc

    df = pd.read_csv(ns.results) if ns.results.suffix == ".csv" else pd.read_parquet(ns.results)
    comp = compare_methods(
        df,
        metric=ns.metric,
        kind=ns.kind,
        higher_is_better=not ns.lower_is_better,
        n_bootstrap=ns.bootstrap_n,
    )
    if comp.empty:
        print(f"[compare] no rows with kind={ns.kind} metric={ns.metric}", flush=True)
        return 1
    pd.set_option("display.float_format", lambda v: f"{v: .4f}")
    print(comp.to_string(index=False), flush=True)
    if ns.out is not None:
        ns.out.parent.mkdir(parents=True, exist_ok=True)
        comp.to_csv(ns.out, index=False)
        print(f"[compare] wrote {ns.out}", flush=True)
    return 0
