"""``neuroco-probe`` CLI + markdown renderer (core).

Loads a run produced by `neuroco train` (`metrics.json` + `best.pt`),
rebuilds env + policy via `neuro_co.core.factory`, samples a fresh batch,
fits the encoder probes, runs PCA / ICA / t-SNE, writes ``probes.json`` +
``probes.md`` + ``figures/*.png``.

The math lives in `neuro_co.probe.probes` / `neuro_co.probe.discovered`;
this module is glue + reporting only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neuro_co.probe.probes import ProbeResult, encoder_layer_count, fit_concept_probes


def _load_run_meta(run_dir: Any) -> dict[str, Any]:
    """Read `<run_dir>/metrics.json` -> flattened training args (problem + arch)."""
    p = Path(run_dir) / "metrics.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"Expected metrics.json at {p}; pass a directory produced by `neuroco train`."
        )
    args = json.loads(p.read_text()).get("args", {})
    arch = {k: args[k] for k in ("backbone", "hidden_dim", "num_layers", "num_heads") if k in args}
    return {"problem": args.get("problem"), "size": args.get("size", 50), "arch": arch}


def _resolve_ckpt(run_dir: Any, override: str | None) -> str:
    if override:
        return str(override)
    for name in ("best.pt", "latest.pt"):
        cand = Path(run_dir) / name
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(f"No best.pt / latest.pt under {run_dir}; pass --ckpt explicitly.")


def _alignment_payload(d: Any) -> dict[str, Any]:
    return {
        "method": d.method,
        "n_components": d.n_components,
        "embed_dim": d.embed_dim,
        "explained_variance_ratio": d.explained_variance_ratio,
        "alignment": [
            {
                "direction_index": a.direction_index,
                "explained_variance_ratio": a.explained_variance_ratio,
                "best_concept": a.best_concept,
                "best_abs_pearson": a.best_abs_pearson,
                "pearson_per_concept": a.pearson_per_concept,
            }
            for a in d.alignment
        ],
    }


def _render_probe_markdown(
    results: list[ProbeResult],
    meta: dict[str, Any],
    *,
    figures: dict[str, Any] | None = None,
    pca_summary: dict[str, Any] | None = None,
    ica_summary: dict[str, Any] | None = None,
) -> str:
    parts: list[str] = ["# Encoder-probe report\n\n"]
    parts.append("## Configuration\n\n")
    parts.append(
        "| Key | Value |\n|---|---|\n"
        f"| ckpt | `{meta.get('ckpt_path')}` |\n"
        f"| env | `{meta.get('env')}` |\n"
        f"| model | `{meta.get('model')}` |\n"
        f"| num_instances | {meta.get('num_instances')} |\n"
        f"| epochs | {meta.get('epochs')} |\n"
        f"| val_frac | {meta.get('val_frac')} |\n"
        f"| lr | {meta.get('lr')} |\n"
        f"| pca_components | {meta.get('pca_components')} |\n"
    )

    parts.append("\n## Probes\n\n")
    if not results:
        parts.append("No probes ran because the environment supplied no matching concept labels.\n")
        return "".join(parts)

    # Cross-layer overview heatmap up front.
    if figures and figures.get("layer_concept_heatmap"):
        parts.append("### Layer x concept val_acc heatmap\n\n")
        parts.append(f"![layer x concept heatmap]({figures['layer_concept_heatmap']})\n\n")

    # Group by layer; sort so the final output (`-1`) is rendered last.
    by_layer: dict[int, list[ProbeResult]] = {}
    for r in results:
        by_layer.setdefault(int(r.layer), []).append(r)
    layer_order = sorted(by_layer, key=lambda x: (x == -1, x))

    for layer in layer_order:
        label = "final encoder output" if layer == -1 else f"encoder layer {layer}"
        parts.append(f"\n### {label}\n\n")
        parts.append(
            "| Concept | acc | bal. acc | F1 | ROC-AUC | precision | recall | pos. frac. |\n"
            "|---|---|---|---|---|---|---|---|\n"
        )
        for r in by_layer[layer]:
            parts.append(
                f"| `{r.concept}` | {r.val_acc:.3f} | {r.val_balanced_acc:.3f} | "
                f"{r.val_f1:.3f} | "
                f"{r.val_roc_auc:.3f} | {r.val_precision:.3f} | {r.val_recall:.3f} | "
                f"{r.val_positive_fraction:.3f} |\n"
            )
        parts.append("\n_Confusion matrices (val split):_\n\n")
        for r in by_layer[layer]:
            cm = r.confusion_matrix
            parts.append(
                f"- `{r.concept}`: TN={cm[0][0]}, FP={cm[0][1]}, "
                f"FN={cm[1][0]}, TP={cm[1][1]} (n_val={r.n_val})\n"
            )
        if figures:
            metrics_key = f"probe_metrics_layer_{layer}"
            roc_key = f"probe_roc_layer_{layer}"
            if figures.get(metrics_key):
                parts.append(f"\n![{label} probe metrics]({figures[metrics_key]})\n")
            if figures.get(roc_key):
                parts.append(f"\n![{label} probe ROC]({figures[roc_key]})\n")

    def _render_directions(
        summary: dict[str, Any],
        *,
        title: str,
        prefix: str,
        variance_fig_key: str,
        heatmap_fig_key: str,
        scatter_fig_key: str,
    ) -> None:
        parts.append(f"\n## Discovered directions ({title})\n\n")
        parts.append(
            "| dir | variance share | best concept | abs. Pearson r |\n|---|---|---|---|\n"
        )
        for entry in summary["alignment"]:
            parts.append(
                f"| {entry['direction_index']} | "
                f"{entry['explained_variance_ratio']:.3f} | "
                f"{entry.get('best_concept') or '-'} | "
                f"{entry['best_abs_pearson']:.3f} |\n"
            )
        if figures and figures.get(variance_fig_key):
            parts.append(f"\n![{title} variance]({figures[variance_fig_key]})\n")
        if figures and figures.get(heatmap_fig_key):
            parts.append(
                f"\n![{title} direction-concept Pearson r heatmap]({figures[heatmap_fig_key]})\n"
            )
        scatter_figs = (figures or {}).get(scatter_fig_key, {})
        if scatter_figs:
            parts.append(f"\n### {prefix}1/{prefix}2 scatter per concept\n")
            for concept, p in scatter_figs.items():
                parts.append(f"\n![{title} `{concept}`]({p})\n")

    if pca_summary:
        _render_directions(
            pca_summary,
            title="PCA",
            prefix="PC",
            variance_fig_key="explained_variance",
            heatmap_fig_key="pca_heatmap",
            scatter_fig_key="pca_scatter",
        )

    if ica_summary:
        _render_directions(
            ica_summary,
            title="ICA",
            prefix="IC",
            variance_fig_key="ica_variance",
            heatmap_fig_key="ica_heatmap",
            scatter_fig_key="ica_scatter",
        )

    tsne_figs = (figures or {}).get("tsne_scatter", {})
    if tsne_figs:
        parts.append("\n## t-SNE projection\n")
        parts.append(
            f"_2-D t-SNE of encoder embeddings (perplexity={meta.get('tsne_perplexity', '?')})._\n"
        )
        for concept, p in tsne_figs.items():
            parts.append(f"\n![t-SNE `{concept}`]({p})\n")

    parts.append(
        "\n_High val accuracy (>> 0.5) means the encoder already linearly "
        "separates the concept. Look at the discovered-direction table to see "
        "whether an unsupervised axis of variance already captures the concept._\n"
    )
    return "".join(parts)


def _save(ax: Any, path: Path) -> None:
    """Tight-layout, save to `path` at 150 dpi, close the figure."""
    import matplotlib.pyplot as plt

    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)


def _render_figures(
    results: list[ProbeResult],
    disc_pca: Any,
    disc_ica: Any,
    pca_summary: dict[str, Any] | None,
    ica_summary: dict[str, Any] | None,
    *,
    tsne: bool,
    tsne_perplexity: float,
    seed: int,
    figures_dir: Path,
    model: Any,
    env: Any,
    td: Any,
    concepts: Any,
) -> dict[str, Any]:
    """Generate all PNGs and return a `figures` dict for the markdown."""
    import numpy as _np

    from neuro_co.probe import viz

    figures: dict[str, Any] = {}

    if results:
        # Group by encoder layer for per-layer panels.
        by_layer: dict[int, list[ProbeResult]] = {}
        for r in results:
            by_layer.setdefault(int(r.layer), []).append(r)
        for layer, layer_results in by_layer.items():
            tag = "final" if layer == -1 else f"layer_{layer}"
            _save(
                viz.plot_probe_metrics_bars(layer_results),
                figures_dir / f"probe_metrics_{tag}.png",
            )
            figures[f"probe_metrics_layer_{layer}"] = f"figures/probe_metrics_{tag}.png"
            try:
                _save(
                    viz.plot_probe_roc(layer_results),
                    figures_dir / f"probe_roc_{tag}.png",
                )
                figures[f"probe_roc_layer_{layer}"] = f"figures/probe_roc_{tag}.png"
            except Exception:
                pass

        # Cross-layer overview heatmap (layer x concept val_acc).
        if len(by_layer) > 1:
            _save(
                viz.plot_layer_concept_heatmap(results, metric="val_acc"),
                figures_dir / "layer_concept_heatmap.png",
            )
            figures["layer_concept_heatmap"] = "figures/layer_concept_heatmap.png"

    if disc_pca is not None and pca_summary is not None:
        _save(
            viz.plot_explained_variance(disc_pca.explained_variance_ratio),
            figures_dir / "explained_variance.png",
        )
        figures["explained_variance"] = "figures/explained_variance.png"
        _save(
            viz.plot_direction_concept_heatmap(pca_summary["alignment"], method_label="PC"),
            figures_dir / "pca_heatmap.png",
        )
        figures["pca_heatmap"] = "figures/pca_heatmap.png"
        pca_scores = _np.asarray(disc_pca.scores)
        figures["pca_scatter"] = {}
        for concept, labels_arr in disc_pca.concept_labels.items():
            _save(
                viz.plot_pca_embeddings(pca_scores, _np.asarray(labels_arr), concept=concept),
                figures_dir / f"pca_{concept}.png",
            )
            figures["pca_scatter"][concept] = f"figures/pca_{concept}.png"

    if disc_ica is not None and ica_summary is not None:
        ax = viz.plot_explained_variance(disc_ica.explained_variance_ratio)
        ax.set_title("ICA per-direction score-variance share")
        _save(ax, figures_dir / "ica_variance.png")
        figures["ica_variance"] = "figures/ica_variance.png"
        _save(
            viz.plot_direction_concept_heatmap(ica_summary["alignment"], method_label="IC"),
            figures_dir / "ica_heatmap.png",
        )
        figures["ica_heatmap"] = "figures/ica_heatmap.png"
        ica_scores = _np.asarray(disc_ica.scores)
        figures["ica_scatter"] = {}
        for concept, labels_arr in disc_ica.concept_labels.items():
            _save(
                viz.plot_scatter_2d_by_label(
                    ica_scores,
                    _np.asarray(labels_arr),
                    title=f"ICA - colored by `{concept}`",
                    xlabel="IC1",
                    ylabel="IC2",
                ),
                figures_dir / f"ica_{concept}.png",
            )
            figures["ica_scatter"][concept] = f"figures/ica_{concept}.png"

    if tsne:
        from neuro_co.probe.discovered import discover_directions, fit_tsne

        base = disc_pca or disc_ica
        if base is None:
            base = discover_directions(
                model, env, td, concepts=concepts, n_components=2, method="pca"
            )
        tsne_coords = fit_tsne(
            base.embeddings,
            perplexity=float(tsne_perplexity),
            random_state=int(seed),
        )
        figures["tsne_scatter"] = {}
        for concept, labels_arr in base.concept_labels.items():
            _save(
                viz.plot_tsne_embeddings(tsne_coords, _np.asarray(labels_arr), concept=concept),
                figures_dir / f"tsne_{concept}.png",
            )
            figures["tsne_scatter"][concept] = f"figures/tsne_{concept}.png"

    return figures


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint: ``neuroco probe <run_dir> [opts]``."""
    import argparse

    import torch

    from neuro_co.core.factory import make_env, make_model
    from neuro_co.probe.discovered import discover_directions

    parser = argparse.ArgumentParser(
        description="Train linear encoder probes on a trained neuro-co policy."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory of a prior `neuroco train` run (must contain `metrics.json` + `best.pt`).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to the checkpoint to load (default: <run_dir>/best.pt or latest.pt).",
    )
    parser.add_argument(
        "--num-instances", type=int, default=16, help="Batch size sampled from the env."
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--val-frac", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <run_dir>/probes).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--layer-indices",
        type=str,
        nargs="*",
        default=["all"],
        help="Probe these intermediate encoder layers. "
        "Accepts integers (`--layer-indices 0 1 2`), `all` (default: every "
        "encoder layer), or `none` (final output only).",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=10,
        help="PCA components for the discovered-direction analysis (0 disables).",
    )
    parser.add_argument(
        "--ica-components",
        type=int,
        default=6,
        help="ICA components for the discovered-direction analysis (0 disables).",
    )
    parser.add_argument(
        "--tsne",
        action="store_true",
        help="Compute a 2-D t-SNE projection of encoder embeddings and "
        "render a per-concept scatter for each default concept.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="Perplexity for the t-SNE projection (clamped if `>= n_samples`).",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip PNG generation (just write JSON + markdown text).",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default=None,
        help=(
            "Concept-bank key to resolve from `neuro_co.xai.concept_registry` "
            "('cvrptw', 'op', 'fjsp', …). If omitted, read from the run dir's "
            "`metrics.json` (`args.problem`)."
        ),
    )
    args = parser.parse_args(argv)

    out_dir = args.out or (args.run_dir / "probes")
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    if not args.no_figures:
        figures_dir.mkdir(exist_ok=True)

    meta = _load_run_meta(args.run_dir)
    env = make_env(str(meta["problem"]), size=int(meta.get("size", 50)))
    model = make_model(env, **meta.get("arch", {}))

    ckpt_path = _resolve_ckpt(args.run_dir, args.ckpt)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state), strict=False)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model = model.to(device).eval()
    td = env.reset(args.num_instances, generator=torch.Generator().manual_seed(args.seed)).to(
        device
    )

    # Resolve `--layer-indices` (accepts integers, "all", or "none").
    raw_layers = args.layer_indices or []
    if any(s == "none" for s in raw_layers):
        layer_indices: list[int] | None = None
    elif any(s == "all" for s in raw_layers):
        n_layers = encoder_layer_count(model)
        layer_indices = list(range(n_layers)) if n_layers > 0 else None
    else:
        try:
            layer_indices = [int(s) for s in raw_layers] or None
        except ValueError as e:
            raise SystemExit(
                f"--layer-indices: expected integers or 'all'/'none', got {raw_layers!r}"
            ) from e

    # Resolve the problem's concept bank from the core registry.
    from neuro_co.core.concepts import concept_registry
    from neuro_co.problems import load_plugins

    load_plugins()  # discover neuro_co.problems plug-ins -> populate registry
    problem = args.problem or meta["problem"]
    if problem is None:
        raise SystemExit("could not resolve problem from run dir; pass --problem explicitly")
    if not concept_registry.names():
        raise SystemExit(
            "no concept banks registered. Install `neuro-co-problems` (or another "
            "package declaring a `neuro_co.problems` entry point) to populate the "
            "concept registry."
        )
    bank = concept_registry.get(str(problem).lower())

    results = fit_concept_probes(
        model,
        env,
        td,
        concepts=bank.concepts,
        val_frac=args.val_frac,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        layer_indices=layer_indices,
    )

    meta = {
        "ckpt_path": str(ckpt_path),
        "problem": str(problem),
        "num_instances": int(args.num_instances),
        "epochs": int(args.epochs),
        "val_frac": float(args.val_frac),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "pca_components": int(args.pca_components),
        "ica_components": int(args.ica_components),
        "tsne": bool(args.tsne),
        "tsne_perplexity": float(args.tsne_perplexity),
    }

    # Discovered directions (PCA + ICA each optional).
    pca_summary: dict[str, Any] | None = None
    ica_summary: dict[str, Any] | None = None
    disc_pca = None
    disc_ica = None
    if args.pca_components > 0:
        disc_pca = discover_directions(
            model,
            env,
            td,
            concepts=bank.concepts,
            n_components=int(args.pca_components),
            method="pca",
        )
        pca_summary = _alignment_payload(disc_pca)
    if args.ica_components > 0:
        disc_ica = discover_directions(
            model,
            env,
            td,
            concepts=bank.concepts,
            n_components=int(args.ica_components),
            method="ica",
            random_state=int(args.seed),
        )
        ica_summary = _alignment_payload(disc_ica)

    figures: dict[str, Any] = {}
    if not args.no_figures:
        figures = _render_figures(
            results,
            disc_pca,
            disc_ica,
            pca_summary,
            ica_summary,
            tsne=bool(args.tsne),
            tsne_perplexity=float(args.tsne_perplexity),
            seed=int(args.seed),
            figures_dir=figures_dir,
            model=model,
            env=env,
            td=td,
            concepts=bank.concepts,
        )

    payload = {
        "config": meta,
        "probes": [
            {
                "concept": r.concept,
                "layer": r.layer,
                "train_acc": r.train_acc,
                "val_acc": r.val_acc,
                "val_balanced_acc": r.val_balanced_acc,
                "val_positive_fraction": r.val_positive_fraction,
                "val_f1": r.val_f1,
                "val_roc_auc": r.val_roc_auc,
                "val_precision": r.val_precision,
                "val_recall": r.val_recall,
                "confusion_matrix": r.confusion_matrix,
                "n_train": r.n_train,
                "n_val": r.n_val,
            }
            for r in results
        ],
        "pca": pca_summary,
        "ica": ica_summary,
    }
    (out_dir / "probes.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "probes.md").write_text(
        _render_probe_markdown(
            results,
            meta,
            figures=figures,
            pca_summary=pca_summary,
            ica_summary=ica_summary,
        )
    )
    print(out_dir / "probes.md")


if __name__ == "__main__":
    main()
