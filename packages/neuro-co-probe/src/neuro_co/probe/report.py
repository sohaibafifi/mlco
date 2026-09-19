"""Markdown + PNG bundle generator for an explanation run.

Reads `explanation.json` (+ optional `instances.pt`) from a core
run directory, produces `report.md` and a `figures/` folder with:

- one route + top-k overlay per instance;
- the per-step deletion-faithfulness flip-rate curve;
- a histogram of the most-attributed nodes;
- aggregate stats: mean / median / std of flip rate, top-k coverage.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from neuro_co.probe.viz import (
    load_artefacts,
    plot_flip_rate,
    plot_route_attribution,
)


def write_explanation_report(
    run_dir: str | Path,
    *,
    max_instances_plotted: int = 4,
    overlay_step: int | None = None,
) -> Path:
    """Generate `report.md` and `figures/*.png` under `run_dir`.

    Parameters
    ----------
    run_dir
        Directory containing `explanation.json` and (optionally)
        `instances.pt`.
    max_instances_plotted
        Cap on number of route overlays generated.
    overlay_step
        If given, plot the route+top-k for that decoding step; if None,
        aggregate over all steps.

    Returns
    -------
    Path to the written `report.md`.
    """
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    bundle = load_artefacts(run_dir)
    explanation: dict[str, Any] = bundle["explanation"]
    instances: dict[str, Any] = bundle["instances"]

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    attrib = explanation["attribution"]
    faith = explanation["faithfulness"]
    actions = np.asarray(attrib["actions"])
    top_nodes = np.asarray(attrib["top_k_nodes"])
    top_scores = np.asarray(attrib["top_k_scores"])
    num_instances = int(explanation["num_instances"])
    num_steps = int(explanation["num_steps"])
    feature_keys = explanation.get("feature_keys", [])

    # Flip-rate curve (deletion).
    fig_flip = run_dir / "figures" / "flip_rate.png"
    ax = plot_flip_rate(faith["per_step_flip_rate"])
    ax.figure.tight_layout()
    ax.figure.savefig(fig_flip, dpi=150)
    plt.close(ax.figure)

    # Sufficiency curve (keep top-k only), if present.
    fig_suff: Path | None = None
    suff_block = faith.get("sufficiency")
    if suff_block is not None and suff_block.get("per_step_keep_rate"):
        fig_suff = figures_dir / "sufficiency.png"
        ax = plot_flip_rate(
            suff_block["per_step_keep_rate"],
            title="Sufficiency - top-k keep rate per step",
        )
        ax.set_ylabel("kept-action rate")
        ax.figure.tight_layout()
        ax.figure.savefig(fig_suff, dpi=150)
        plt.close(ax.figure)

    # Route overlays per instance.
    route_paths: list[Path] = []
    locs_all = instances.get("locs")
    if locs_all is not None:
        locs_all_np = locs_all.numpy() if hasattr(locs_all, "numpy") else np.asarray(locs_all)
        n_plot = min(max_instances_plotted, num_instances)
        for i in range(n_plot):
            fig_path = figures_dir / f"route_instance_{i}.png"
            ax = plot_route_attribution(
                locs=locs_all_np[i],
                actions=actions[i].tolist(),
                top_k_nodes=top_nodes[i],
                top_k_scores=top_scores[i],
                step=overlay_step,
            )
            ax.figure.tight_layout()
            ax.figure.savefig(fig_path, dpi=150)
            plt.close(ax.figure)
            route_paths.append(fig_path)

    # Most-attributed nodes histogram.
    flat = top_nodes.reshape(-1).tolist()
    counter = Counter(int(i) for i in flat if i >= 0)
    most_common = counter.most_common(15)

    # Aggregate flip-rate stats.
    per_step = np.asarray(faith["per_step_flip_rate"], dtype=float)
    flip_stats = {
        "mean": float(per_step.mean()) if per_step.size else 0.0,
        "median": float(np.median(per_step)) if per_step.size else 0.0,
        "std": float(per_step.std()) if per_step.size else 0.0,
        "first_decision_mean": float(per_step[: max(1, per_step.size // 4)].mean())
        if per_step.size
        else 0.0,
        "last_decision_mean": float(per_step[-max(1, per_step.size // 4) :].mean())
        if per_step.size
        else 0.0,
    }

    md = _render_markdown(
        explanation=explanation,
        num_instances=num_instances,
        num_steps=num_steps,
        feature_keys=feature_keys,
        flip_stats=flip_stats,
        most_common=most_common,
        flip_fig=fig_flip.relative_to(run_dir),
        route_figs=[p.relative_to(run_dir) for p in route_paths],
        sufficiency_fig=fig_suff.relative_to(run_dir) if fig_suff is not None else None,
        sufficiency_block=suff_block,
        sanity_block=faith.get("sanity_check"),
        probes=explanation.get("encoder_probes"),
    )
    md_path = run_dir / "report.md"
    md_path.write_text(md)
    return md_path


def _render_markdown(
    *,
    explanation: dict[str, Any],
    num_instances: int,
    num_steps: int,
    feature_keys: list[str],
    flip_stats: dict[str, float],
    most_common: list[tuple[int, int]],
    flip_fig: Path,
    route_figs: list[Path],
    sufficiency_fig: Path | None = None,
    sufficiency_block: dict[str, Any] | None = None,
    sanity_block: dict[str, Any] | None = None,
    probes: list[dict[str, Any]] | None = None,
) -> str:
    parts: list[str] = []
    parts.append("# Explanation report\n\n")
    parts.append("## Configuration\n\n")
    parts.append(
        "| Key | Value |\n|---|---|\n"
        f"| ckpt | `{explanation.get('ckpt_path')}` |\n"
        f"| env | `{explanation.get('env')}` |\n"
        f"| model | `{explanation.get('model')}` |\n"
        f"| num_instances | {num_instances} |\n"
        f"| num_steps | {num_steps} |\n"
        f"| top_k | {explanation.get('top_k')} |\n"
        f"| feature_keys | {feature_keys} |\n"
    )

    parts.append("\n## Faithfulness summary\n\n")
    parts.append(
        "| Statistic | Value |\n|---|---|\n"
        f"| mean flip rate | {flip_stats['mean']:.3f} |\n"
        f"| median | {flip_stats['median']:.3f} |\n"
        f"| std | {flip_stats['std']:.3f} |\n"
        f"| first 25% of steps (mean) | {flip_stats['first_decision_mean']:.3f} |\n"
        f"| last 25% of steps (mean) | {flip_stats['last_decision_mean']:.3f} |\n"
    )
    parts.append(f"\n![flip rate]({flip_fig})\n")

    parts.append("\n## Most-attributed nodes\n\n")
    parts.append("| Node index | # appearances in top-k |\n|---|---|\n")
    for idx, count in most_common:
        parts.append(f"| {idx} | {count} |\n")

    method = explanation.get("method", "gradient")
    parts.append(f"\n_Attribution method: **{method}**_\n")
    if method in {"ig", "integrated_gradients"}:
        mp = explanation.get("method_params", {})
        parts.append(
            f"_IG steps: {mp.get('ig_steps', '?')}, baseline: {mp.get('ig_baseline', '?')}_\n"
        )

    if sufficiency_block is not None:
        parts.append("\n## Sufficiency (keep only top-k)\n\n")
        parts.append(
            "| Statistic | Value |\n|---|---|\n"
            f"| mean keep rate | {sufficiency_block['mean_keep_rate']:.3f} |\n"
            f"| top-k used | {sufficiency_block['top_k_used']} |\n"
            f"| num_steps | {sufficiency_block['num_steps']} |\n"
        )
        if sufficiency_fig is not None:
            parts.append(f"\n![sufficiency]({sufficiency_fig})\n")
        parts.append("\n_High keep rate = top-k alone reproduces the chosen action._\n")

    if sanity_block is not None:
        parts.append("\n## Parameter-randomization sanity check\n\n")
        parts.append(
            "| Statistic | Value |\n|---|---|\n"
            f"| mode | `{sanity_block['mode']}` |\n"
            f"| mean Jaccard (top-k orig vs randomized) | "
            f"{sanity_block['mean_jaccard']:.3f} |\n"
            f"| chance baseline (k / num_nodes) | "
            f"{sanity_block['chance_jaccard']:.3f} |\n"
            f"| num trials | {sanity_block['num_trials']} |\n"
        )
        parts.append(
            "\n_Mean Jaccard close to chance = attribution depends on the "
            "trained weights (sanity passes). Much higher than chance = "
            "explanation is largely insensitive to the model._\n"
        )

    if probes:
        parts.append("\n## Encoder probes\n\n")
        parts.append(
            "| Concept | val acc | val balanced acc | val pos. frac. | n_train | n_val |\n"
            "|---|---|---|---|---|---|\n"
        )
        for p in probes:
            parts.append(
                f"| `{p['concept']}` | {p['val_acc']:.3f} | "
                f"{p['val_balanced_acc']:.3f} | {p['val_positive_fraction']:.3f} | "
                f"{p['n_train']} | {p['n_val']} |\n"
            )
        parts.append(
            "\n_High val accuracy (>> 0.5) means the encoder already "
            "linearly separates the concept._\n"
        )

    if route_figs:
        parts.append("\n## Routes + top-k overlay (per instance)\n")
        for p in route_figs:
            parts.append(f"\n![{p.stem}]({p})\n")
    else:
        parts.append("\n_No `instances.pt` found - route overlays skipped._\n")

    return "".join(parts)


def main() -> None:
    """CLI entrypoint: `neuroco-explain-report <run_dir> [--step N]`."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a markdown + PNG report from an explanation run."
    )
    parser.add_argument("run_dir", type=Path, help="Directory holding explanation.json")
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="If given, route overlays show top-k for that decoding step; "
        "otherwise aggregate over all steps.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=4,
        help="Maximum number of route overlay PNGs to generate.",
    )
    args = parser.parse_args()
    out = write_explanation_report(
        args.run_dir,
        max_instances_plotted=args.max_instances,
        overlay_step=args.step,
    )
    # Print to stdout so the shell user gets the path.
    print(out)


if __name__ == "__main__":
    main()
