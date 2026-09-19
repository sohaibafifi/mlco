"""Standardised plotting harness for `neuroco results` parquet files.

Four plots generated from the long-format DataFrame
emitted by `neuro_co.cli.results.collect`. PDF output;
PNG also supported by extension.

Functions:

- `plot_probe_accuracy_heatmap(df, problem)`: layer x concept
  `val_acc` heatmap, one panel per problem.
- `plot_attribution_quality(df)`: deletion + sufficiency per
  method, grouped bars across problems.
- `plot_sanity_pass_rate(df)`: `sanity.mean_jaccard` per method,
  with the chance-baseline reference line.
- `plot_cross_seed_stability(df)`: std of probe `val_acc` across
  seeds, heatmap per problem.

CLI: `neuroco figures <results.parquet> --out figures/` writes all
four into the output dir. Missing data for a given panel means that
panel is skipped (a warning is printed), the rest still renders.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# Defaults for a conference column (8-11 cm wide). Plotting functions
# apply these settings; importing the module leaves Matplotlib unchanged.
_CAMERA_READY_RC: dict[str, Any] = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,  # embed TrueType fonts
    "ps.fonttype": 42,
}


def _apply_rc() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(_CAMERA_READY_RC)


def _save(fig: Any, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Experiment plots.
# ---------------------------------------------------------------------------


def plot_probe_accuracy_heatmap(df: Any, out: Path) -> Path | None:
    """Layer x concept val_acc heatmap, one panel per problem.

    Returns the written path, or `None` if the DataFrame has no probe
    rows.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    _apply_rc()
    probes = df[(df["kind"] == "probe") & (df["metric"] == "val_acc")]
    if probes.empty:
        log.warning("figures: no probe rows; skipping probe-accuracy heatmap")
        return None
    # Average over seeds.
    pivot = probes.groupby(["problem", "layer", "concept"])["value"].mean().reset_index()
    problems = sorted(pivot["problem"].unique())
    n = len(problems)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 2.6), squeeze=False)
    for ax, problem in zip(axes[0], problems, strict=False):
        sub = pivot[pivot["problem"] == problem]
        layers = sorted(sub["layer"].unique())
        concepts = sorted(sub["concept"].unique())
        grid = np.full((len(layers), len(concepts)), np.nan)
        for _, row in sub.iterrows():
            i = layers.index(int(row["layer"]))
            j = concepts.index(str(row["concept"]))
            grid[i, j] = row["value"]
        im = ax.imshow(grid, aspect="auto", vmin=0.5, vmax=1.0, cmap="viridis")
        ax.set_title(problem)
        ax.set_xticks(range(len(concepts)))
        ax.set_xticklabels(concepts, rotation=30, ha="right")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(["final" if lyr == -1 else f"L{lyr}" for lyr in layers])
        ax.set_xlabel("concept")
        if ax is axes[0, 0]:
            ax.set_ylabel("encoder layer")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not np.isnan(grid[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{grid[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if grid[i, j] < 0.8 else "black",
                    )
    fig.colorbar(im, ax=axes[0, -1], fraction=0.046, pad=0.04, label="val_acc")
    return _save(fig, out)


def plot_attribution_quality(df: Any, out: Path) -> Path | None:
    """Deletion mean-flip-rate + sufficiency mean-keep-rate per method.

    Grouped-bar chart, X = method, hue = problem, panels = metric.
    Higher flip-rate = more faithful; higher keep-rate = more
    sufficient.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    _apply_rc()
    expl = df[df["kind"] == "explain"]
    if expl.empty:
        log.warning("figures: no explain rows; skipping attribution-quality plot")
        return None
    metrics = ["deletion.mean_flip_rate", "sufficiency.mean_keep_rate"]
    expl = expl[expl["metric"].isin(metrics)]
    if expl.empty:
        log.warning("figures: no deletion/sufficiency rows; skipping attribution-quality plot")
        return None
    agg = expl.groupby(["metric", "problem", "method"])["value"].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)
    for ax, metric in zip(axes, metrics, strict=False):
        sub = agg[agg["metric"] == metric]
        problems = sorted(sub["problem"].unique())
        methods = sorted(sub["method"].unique())
        x = np.arange(len(problems))
        width = 0.8 / max(1, len(methods))
        for i, method in enumerate(methods):
            vals = [
                float(sub[(sub["problem"] == p) & (sub["method"] == method)]["value"].mean())
                if not sub[(sub["problem"] == p) & (sub["method"] == method)].empty
                else np.nan
                for p in problems
            ]
            ax.bar(x + (i - len(methods) / 2 + 0.5) * width, vals, width=width, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(problems, rotation=20, ha="right")
        ax.set_ylabel("rate")
        ax.set_title(metric.split(".")[-1].replace("_", " "))
        ax.set_ylim(0, 1)
    axes[-1].legend(loc="upper right", title="method")
    return _save(fig, out)


def plot_sanity_pass_rate(df: Any, out: Path) -> Path | None:
    """`sanity.mean_jaccard` per method with the chance baseline marker."""
    import matplotlib.pyplot as plt
    import numpy as np

    _apply_rc()
    sanity = df[(df["kind"] == "explain") & (df["metric"] == "sanity.mean_jaccard")]
    chance = df[(df["kind"] == "explain") & (df["metric"] == "sanity.chance_jaccard")]
    if sanity.empty:
        log.warning("figures: no sanity rows; skipping sanity-pass plot")
        return None
    agg = sanity.groupby(["problem", "method"])["value"].mean().reset_index()
    chance_per_problem = chance.groupby("problem")["value"].mean()
    problems = sorted(agg["problem"].unique())
    methods = sorted(agg["method"].unique())
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    x = np.arange(len(problems))
    width = 0.8 / max(1, len(methods))
    for i, method in enumerate(methods):
        vals = [
            float(agg[(agg["problem"] == p) & (agg["method"] == method)]["value"].mean())
            if not agg[(agg["problem"] == p) & (agg["method"] == method)].empty
            else np.nan
            for p in problems
        ]
        ax.bar(x + (i - len(methods) / 2 + 0.5) * width, vals, width=width, label=method)
    for j, p in enumerate(problems):
        c = chance_per_problem.get(p, np.nan)
        if not np.isnan(c):
            ax.hlines(
                c,
                xmin=x[j] - 0.4,
                xmax=x[j] + 0.4,
                colors="black",
                linestyles="--",
                linewidth=0.8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(problems, rotation=20, ha="right")
    ax.set_ylabel("sanity Jaccard")
    ax.set_title("Adebayo sanity vs chance (dashed)")
    ax.legend(loc="upper right", title="method", fontsize=7)
    return _save(fig, out)


def plot_cross_seed_stability(df: Any, out: Path) -> Path | None:
    """Per-(layer, concept) std of probe val_acc across seeds, heatmap."""
    import matplotlib.pyplot as plt
    import numpy as np

    _apply_rc()
    probes = df[(df["kind"] == "probe") & (df["metric"] == "val_acc")]
    if probes.empty:
        log.warning("figures: no probe rows; skipping cross-seed stability plot")
        return None
    if probes["seed"].nunique() < 2:
        log.warning("figures: only one seed in probe rows; skipping cross-seed plot")
        return None
    stds = probes.groupby(["problem", "layer", "concept"])["value"].std(ddof=0).reset_index()
    problems = sorted(stds["problem"].unique())
    n = len(problems)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 2.6), squeeze=False)
    for ax, problem in zip(axes[0], problems, strict=False):
        sub = stds[stds["problem"] == problem]
        layers = sorted(sub["layer"].unique())
        concepts = sorted(sub["concept"].unique())
        grid = np.full((len(layers), len(concepts)), np.nan)
        for _, row in sub.iterrows():
            i = layers.index(int(row["layer"]))
            j = concepts.index(str(row["concept"]))
            grid[i, j] = row["value"]
        im = ax.imshow(grid, aspect="auto", cmap="magma")
        ax.set_title(problem)
        ax.set_xticks(range(len(concepts)))
        ax.set_xticklabels(concepts, rotation=30, ha="right")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(["final" if lyr == -1 else f"L{lyr}" for lyr in layers])
        if ax is axes[0, 0]:
            ax.set_ylabel("encoder layer")
    fig.colorbar(im, ax=axes[0, -1], fraction=0.046, pad=0.04, label="std(val_acc)")
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Bulk driver.
# ---------------------------------------------------------------------------


def render_all(df: Any, out_dir: Path, ext: str = "pdf") -> list[Path]:
    """Render each available plot into `out_dir`. Skipped panels are listed in the log."""
    out_dir = Path(out_dir)
    written: list[Path] = []
    panels = (
        ("probe_accuracy_heatmap", plot_probe_accuracy_heatmap),
        ("attribution_quality", plot_attribution_quality),
        ("sanity_pass_rate", plot_sanity_pass_rate),
        ("cross_seed_stability", plot_cross_seed_stability),
    )
    for name, fn in panels:
        p = fn(df, out_dir / f"{name}.{ext}")
        if p is not None:
            written.append(p)
    return written


def main_cli(argv: list[str] | None = None) -> int:
    """`neuroco figures <results.parquet> --out figures/` entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(prog="neuroco figures")
    parser.add_argument(
        "results",
        type=Path,
        help="Path to a results.parquet or .csv produced by `neuroco results`.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures"),
        help="Output directory (created if missing). Default `figures/`.",
    )
    parser.add_argument(
        "--ext",
        choices=("pdf", "png"),
        default="pdf",
        help="Figure file extension. Default pdf.",
    )
    ns = parser.parse_args(argv)

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pandas is required for `neuroco figures`") from exc

    df = pd.read_csv(ns.results) if ns.results.suffix == ".csv" else pd.read_parquet(ns.results)
    written = render_all(df, ns.out, ext=ns.ext)
    for p in written:
        print(f"[figures] {p}", flush=True)
    if not written:
        print("[figures] no panels produced; check the input parquet", flush=True)
        return 1
    return 0
