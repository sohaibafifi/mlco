"""Plot constraint-attribution scores and counterfactual agreement.

`hero_plot` groups scores by problem and seed, with optional concept-probe
annotations. Other plots summarize multiplier modes and agreement with the
sampled counterfactual reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# Camera-ready defaults.
_RC = {
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,  # embed TrueType -- conference-safe
    "ps.fonttype": 42,
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def hero_plot(parquet: Path, *, out: Path, probes_parquet: Path | None = None) -> Path:
    """Stacked-bar of Lambda-attr per (problem, seed); optional probe annotation.

    `parquet` must be the aggregate written by `neuroco-cax-benchmark`
    (columns: problem, seed, run_dir, constraint, mean_score,
    num_instances, num_steps).
    """
    import pandas as pd

    df = pd.read_parquet(parquet)
    if df.empty:
        raise ValueError(f"empty Lambda-attr parquet: {parquet}")

    plt.rcParams.update(_RC)
    problems = sorted(df.problem.unique())
    n_panels = len(problems) + (1 if probes_parquet is not None else 0)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(3.0 * n_panels, 3.2),
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, problem in zip(axes[: len(problems)], problems, strict=False):
        sub = df[df.problem == problem]
        seeds = sorted(sub.seed.unique())
        constraints = list(dict.fromkeys(sub.constraint))  # preserve declaration order
        x = list(range(len(seeds)))
        bottom = [0.0] * len(seeds)
        for c in constraints:
            csub = sub[sub.constraint == c]
            heights = [float(csub[csub.seed == s].mean_score.mean()) for s in seeds]
            ax.bar(x, heights, bottom=bottom, label=c, width=0.7)
            bottom = [b + h for b, h in zip(bottom, heights, strict=False)]
        ax.set_title(problem)
        ax.set_xticks(x)
        ax.set_xticklabels([f"seed {s}" for s in seeds])
        ax.set_ylabel(r"$\Lambda_k$ score")
        ax.legend(loc="upper right", framealpha=0.9)

    # Optional probe annotation panel: balanced-acc per concept,
    # averaged over seeds.
    if probes_parquet is not None:
        pdf = pd.read_parquet(probes_parquet)
        psub = pdf[pdf.metric == "val_balanced_acc"]
        ax = axes[-1]
        if not psub.empty:
            tbl = psub.groupby(["problem", "concept"])["value"].mean().unstack()
            tbl = tbl.round(3)
            ax.axis("off")
            ax.set_title("Probe balanced-acc")
            cell_text = [[problem] + [str(v) for v in row] for problem, row in tbl.iterrows()]
            ax.table(
                cellText=cell_text,
                colLabels=["problem", *list(tbl.columns)],
                loc="center",
            )

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def compare_modes_table(
    parquets: dict[str, Path],
    *,
    out: Path,
    problems: list[str] | None = None,
) -> Path:
    """Side-by-side ablation: per-constraint bars per (problem, mode).

    `parquets` keys are mode labels (`'proxy'`, `'lp'`, `'subgrad'`),
    values point at the aggregate parquets produced by
    `neuroco-cax-benchmark --multipliers <mode>`. Produces a row
    of subplots (one per problem) with grouped bars: one cluster
    per constraint family, one bar per mode in the cluster. The
    Spearman rank-correlation of the per-constraint mean_score
    between every mode pair is reported in the subplot title --
    low correlation = modes disagree (= ablation finding).
    """
    import pandas as pd

    if not parquets:
        raise ValueError("parquets dict must be non-empty")
    plt.rcParams.update(_RC)

    # Load + tag every parquet with its mode label.
    frames = []
    for mode, path in parquets.items():
        df = pd.read_parquet(path).copy()
        df["mode"] = mode
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)

    if problems is None:
        problems = sorted(full.problem.unique())
    n = len(problems)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4), constrained_layout=True)
    if n == 1:
        axes = [axes]

    modes = list(parquets.keys())
    width = 0.8 / max(1, len(modes))

    for ax, problem in zip(axes, problems, strict=False):
        sub = full[full.problem == problem]
        if sub.empty:
            ax.set_title(f"{problem} (no data)")
            ax.axis("off")
            continue
        # Aggregate over (seed, run_dir): mean per (mode, constraint).
        agg = sub.groupby(["mode", "constraint"])["mean_score"].mean().unstack("mode")
        constraints = list(agg.index)
        x = list(range(len(constraints)))
        for m_idx, mode in enumerate(modes):
            if mode not in agg.columns:
                continue
            offset = (m_idx - (len(modes) - 1) / 2) * width
            heights = [float(agg.loc[c, mode]) if c in agg.index else 0.0 for c in constraints]
            ax.bar(
                [xi + offset for xi in x],
                heights,
                width=width,
                label=mode,
            )
        # Spearman rank correlation across modes.
        rho_pairs = []
        if len(modes) >= 2 and len(constraints) >= 2:
            for i in range(len(modes)):
                for j in range(i + 1, len(modes)):
                    a = agg[modes[i]].rank().to_numpy()
                    b = agg[modes[j]].rank().to_numpy()
                    rho = float(
                        ((a - a.mean()) * (b - b.mean())).sum()
                        / (
                            ((a - a.mean()) ** 2).sum() ** 0.5 * ((b - b.mean()) ** 2).sum() ** 0.5
                            + 1e-12
                        )
                    )
                    rho_pairs.append(f"{modes[i][0]}/{modes[j][0]}={rho:+.2f}")
        title = problem + (f"   rho:{', '.join(rho_pairs)}" if rho_pairs else "")
        ax.set_title(title, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(constraints, rotation=20, ha="right")
        ax.set_ylabel(r"mean $\Lambda_k$")
        ax.legend(loc="upper right", framealpha=0.9)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def adjudication_plot(
    parquet: Path,
    *,
    out: Path,
    modes: tuple[str, ...] = ("proxy", "lp", "subgrad"),
) -> Path:
    """Per-(problem, mode) mean agreement with the sampled counterfactual reference.

    Bars = mean over seeds; error bars = bootstrap 95% CI on the
    per-step paired diff vs the first mode. Stars on bars whose
    diff CI excludes zero (= significantly different from proxy).
    """
    import pandas as pd

    from neuro_co.cax.adjudicate import (
        bootstrap_ci_diff,
        load_pairwise_matches,
        mcnemar_pvalue,
    )

    df = pd.read_parquet(parquet)
    if df.empty:
        raise ValueError(f"empty adjudication parquet: {parquet}")
    plt.rcParams.update(_RC)

    problems = sorted(df.problem.unique())
    fig, axes = plt.subplots(
        1, len(problems), figsize=(3.4 * len(problems), 3.6), constrained_layout=True
    )
    if len(problems) == 1:
        axes = [axes]

    for ax, problem in zip(axes, problems, strict=False):
        sub = df[df.problem == problem]
        run_dirs = [Path(r) for r in sorted(sub.run_dir.unique())]
        if not run_dirs:
            ax.axis("off")
            ax.set_title(f"{problem} (no data)")
            continue
        try:
            pw = load_pairwise_matches(run_dirs, modes=modes)
        except (FileNotFoundError, KeyError) as exc:
            ax.axis("off")
            ax.set_title(f"{problem} (load fail: {exc})")
            continue
        # No CF coverage -> empty pairs.
        if not pw[modes[0]]:
            ax.axis("off")
            ax.set_title(f"{problem}\n(no CF flips)")
            continue

        means = [sum(pw[m]) / len(pw[m]) for m in modes]
        ax.bar(range(len(modes)), means, color=["#888", "#1f77b4", "#d62728"][: len(modes)])
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels(modes)
        ax.set_ylabel("agreement with counterfactual reference")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{problem}  (n_flips={len(pw[modes[0]])})")

        # Annotate vs first mode (proxy baseline) with bootstrap CI + McNemar p.
        base = pw[modes[0]]
        for i, m in enumerate(modes[1:], start=1):
            if not pw[m]:
                continue
            point, lo, hi = bootstrap_ci_diff(pw[m], base, n_resamples=2000, seed=0)
            _b01, _b10, p = mcnemar_pvalue(pw[m], base)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            ax.annotate(
                f"diff={point:+.3f}\n[{lo:+.3f},{hi:+.3f}] {sig}",
                xy=(i, means[i]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="neuroco-cax-figures")
    sub = parser.add_subparsers(dest="cmd")

    p_hero = sub.add_parser("hero", help="hero plot (default if no subcommand)")
    p_hero.add_argument("parquet", type=Path, help="aggregate Lambda-attr parquet")
    p_hero.add_argument("--out", type=Path, required=True)
    p_hero.add_argument("--probes", type=Path, default=None)

    p_cmp = sub.add_parser("compare", help="ablation: per-mode bars + rank correlation")
    p_cmp.add_argument(
        "--mode",
        action="append",
        nargs=2,
        metavar=("LABEL", "PARQUET"),
        required=True,
        help=(
            "Repeatable. Pairs of (label, parquet) for each backend. "
            "Example: --mode proxy proxy.parquet --mode lp lp.parquet."
        ),
    )
    p_cmp.add_argument("--out", type=Path, required=True)

    p_adj = sub.add_parser(
        "adjudicate",
        help="agreement with the sampled counterfactual reference (bars + bootstrap CI)",
    )
    p_adj.add_argument("parquet", type=Path, help="adjudicate.parquet")
    p_adj.add_argument("--out", type=Path, required=True)
    p_adj.add_argument("--modes", nargs="+", default=["proxy", "lp", "subgrad"])

    args = parser.parse_args(argv)

    if args.cmd == "compare":
        parquets = {label: Path(path) for label, path in args.mode}
        out = compare_modes_table(parquets, out=args.out)
    elif args.cmd == "adjudicate":
        out = adjudication_plot(args.parquet, out=args.out, modes=tuple(args.modes))
    elif args.cmd == "hero":
        out = hero_plot(args.parquet, out=args.out, probes_parquet=args.probes)
    else:
        parser.error("subcommand required: hero | compare | adjudicate (see --help for each)")
    print(f"[cax-figures] wrote {out}")
    return 0


_ = Any  # silence lint when only type-hint use exists below
